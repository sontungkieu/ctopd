"""Offline local-snapshot loading and production-tokenization reuse."""
import json
import hashlib
import copy
import subprocess
from pathlib import Path
import torch


def source_identity():
    root=Path(__file__).resolve().parents[2]
    commit=dirty=None
    if (root/'.git').exists():
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        dirty=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip())
    manifest=root/'docs/H1B_SOURCE.json'
    return dict(commit=commit,dirty=dirty,
                source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())


def snapshot_identity(path):
    root=Path(path).resolve()
    return dict(path=str(root),metadata_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json') if (root/name).is_file()},
        qualification='Metadata identity; individual weight shards not hashed by this helper')


def archive_rows(path,limit=None):
    with open(path) as f:
        for i,line in enumerate(f):
            if limit is not None and i>=limit: break
            yield json.loads(line)


def load_pair(student_path,teacher_path,device):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if str(device).startswith('cuda'):
        if torch.cuda.device_count()!=1 or 'B200' not in torch.cuda.get_device_name(0):
            raise ValueError('Expose exactly one allocated B200 for H1b offline GPU tools')
    models=[]; tokenizers=[]
    for path in (student_path,teacher_path):
        if not Path(path).is_dir(): raise FileNotFoundError('Require local model snapshot: '+str(path))
        tokenizers.append(AutoTokenizer.from_pretrained(path,local_files_only=True))
        models.append(AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
                      torch_dtype=torch.bfloat16).to(device).eval())
        config=getattr(models[-1].config,'text_config',None) or models[-1].config
        if getattr(config,'final_logit_softcapping',None): raise ValueError('H1b softcapped heads unsupported')
    return models,tokenizers


def tokenize_record(record,student_tok,teacher_tok,assume_shared_prompt=False):
    from kdflow.trainer.data_processor.rollout_data_processor import RolloutDataProcessor
    # Reuse production method rather than approximating prompt boundary masks.
    processor=object.__new__(RolloutDataProcessor)
    if 'tea_prompt' not in record and not assume_shared_prompt:
        raise ValueError('Archive lacks tea_prompt; supply enriched H1b archive or explicitly --assume-shared-prompt')
    reason=record.get('finish_reason')
    output=dict(text=record['output'],output_ids=record.get('output_ids',[]),finish_reason=reason)
    if reason is None and not record.get('output_ids'):
        raise ValueError('Cannot reconstruct truncation without finish_reason/output_ids')
    append=not processor._is_length_truncated(output,{'max_new_tokens':8192})
    out={}
    for prefix,tok,prompt in (('stu',student_tok,record['prompt']),('tea',teacher_tok,record.get('tea_prompt',record['prompt']))):
        fields=processor._tokenize_sample(prompt,record['output'],tok,prefix,append_eos=append)
        out.update({k:v[None] for k,v in fields.items() if not k.startswith('_')})
    return out


def row_groups(batch,student_tok,teacher_tok):
    from kdflow.token_alignment import get_tokenizer_byte_codec
    from kdflow.h1b.groups import align_byte_span_groups_batch
    rows=[]
    for key,tok in (('tea',teacher_tok),('stu',student_tok)):
        labels=batch[key+'_input_ids'].roll(-1,1)
        rows.append([tok.convert_ids_to_tokens(labels[i][batch[key+'_loss_mask'][i].bool()].tolist()) for i in range(len(labels))])
    return align_byte_span_groups_batch(*rows,teacher_tok.eos_token,student_tok.eos_token,
        get_tokenizer_byte_codec(teacher_tok).piece_to_bytes,get_tokenizer_byte_codec(student_tok).piece_to_bytes)


def forward_hidden(model,batch,prefix):
    ids=batch[prefix+'_input_ids'].to(next(model.parameters()).device)
    out=model.base_model(ids,attention_mask=batch[prefix+'_attn_mask'].to(ids.device),output_hidden_states=True,use_cache=False)
    return out.hidden_states[-1][batch[prefix+'_loss_mask'].to(ids.device).bool()]


class StudentWrapper(torch.nn.Module):
    def __init__(self,model):
        super().__init__(); self.transformer=model
        # Proxy contains the actual head; avoids assigning model.lm_head to itself.
        import types
        head=model.get_output_embeddings()
        class Head(torch.nn.Module):
            def __init__(self,head): super().__init__(); self.head=head
            @property
            def weight(self): return self.head.weight
            def forward(self,x,skip=False): return x if skip else self.head(x)
        self.model=types.SimpleNamespace(config=model.config,lm_head=Head(head))
    def forward(self,ids,attention_mask=None,**kwargs):
        output=self.transformer.base_model(ids,attention_mask=attention_mask,output_hidden_states=True,use_cache=False)
        return {'hidden_states':output.hidden_states}


class HFPrefixOracle:
    """Branch-safe depth-bounded KV stack for depth-first cover enumeration.

    HF DynamicCache may mutate in place. Each child receives its own copy; the
    parent's cache remains reusable for siblings. Full-prefix fallback bounds
    retained cache depth for unusually long chunks. No branch graph is retained.
    """
    @torch.no_grad()
    def __init__(self,model,prefix_ids,cache_depth=16,use_cache=True):
        if not prefix_ids or cache_depth<1: raise ValueError('Nonempty prefix and positive cache depth required')
        self.model=model; self.prefix=list(prefix_ids); self.depth=cache_depth; self.use_cache=use_cache
        self.device=next(model.parameters()).device
        out=model.base_model(torch.tensor([self.prefix],device=self.device),use_cache=use_cache)
        lp=model.get_output_embeddings()(out.last_hidden_state[0,-1]).float().log_softmax(-1)
        self.stack=[((),lp,getattr(out,'past_key_values',None))]

    @torch.no_grad()
    def __call__(self,branch):
        branch=tuple(branch)
        if not branch: return self.stack[0][1]
        if not self.use_cache or len(branch)>self.depth or self.stack[0][2] is None:
            out=self.model.base_model(torch.tensor([self.prefix+list(branch)],device=self.device),use_cache=False)
            return self.model.get_output_embeddings()(out.last_hidden_state[0,-1]).float().log_softmax(-1)
        common=0
        for level in range(1,min(len(branch),len(self.stack)-1)+1):
            if self.stack[level][0]!=branch[:level]: break
            common=level
        self.stack=self.stack[:common+1]
        for level in range(common+1,len(branch)+1):
            cache=copy.deepcopy(self.stack[-1][2])
            out=self.model.base_model(torch.tensor([[branch[level-1]]],device=self.device),
                attention_mask=torch.ones(1,len(self.prefix)+level,device=self.device,dtype=torch.long),
                past_key_values=cache,use_cache=True)
            lp=self.model.get_output_embeddings()(out.last_hidden_state[0,-1]).float().log_softmax(-1)
            self.stack.append((branch[:level],lp,out.past_key_values))
        return self.stack[-1][1]
