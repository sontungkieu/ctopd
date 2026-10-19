"""Six byte classes times four length bins, stable bucket IDs 0..23."""
import string


def bucket_of(value):
    if not value: raise ValueError('EOS/empty bytes are not mismatch buckets')
    if all(48 <= b <= 57 for b in value): kind=0
    elif value.isspace(): kind=1
    elif any(b>127 for b in value): kind=2
    elif all(65<=b<=90 or 97<=b<=122 for b in value): kind=3
    elif all(chr(b) in string.punctuation for b in value): kind=4
    else: kind=5
    size=0 if len(value)<=2 else 1 if len(value)<=4 else 2 if len(value)<=8 else 3
    return kind*4+size
