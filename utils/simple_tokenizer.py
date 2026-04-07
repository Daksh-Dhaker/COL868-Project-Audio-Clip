# CREDITS: https://github.com/openai/CLIP

import gzip
import html
import os
import shutil
import tempfile
import urllib.request
from functools import lru_cache

import ftfy
import regex as re


BPE_FILENAME = 'bpe_simple_vocab_16e6.txt.gz'
BPE_URL = 'https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz'


def _looks_like_lfs_pointer(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'rb') as handle:
            head = handle.read(120)
        return head.startswith(b'version https://git-lfs.github.com/spec/v1')
    except OSError:
        return False


def _is_valid_gzip(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with gzip.open(path, 'rb') as handle:
            handle.read(2)
        return True
    except OSError:
        return False


def _download_bpe(target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='audioclip_bpe_', suffix='.gz')
    os.close(fd)
    try:
        urllib.request.urlretrieve(BPE_URL, tmp_path)
        if not _is_valid_gzip(tmp_path):
            raise RuntimeError(f'Downloaded BPE file is invalid: {tmp_path}')
        shutil.move(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return target_path


def _resolve_bpe_path() -> str:
    env_path = os.environ.get('AUDIOCLIP_BPE_PATH')
    if env_path:
        if _looks_like_lfs_pointer(env_path):
            raise RuntimeError(
                f'AUDIOCLIP_BPE_PATH points to a Git LFS pointer file: {env_path}. '
                'Please provide a real bpe_simple_vocab_16e6.txt.gz file.'
            )
        if not _is_valid_gzip(env_path):
            raise RuntimeError(
                f'AUDIOCLIP_BPE_PATH is not a valid gzip file: {env_path}. '
                'Please provide a real bpe_simple_vocab_16e6.txt.gz file.'
            )
        return env_path

    repo_bpe = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', BPE_FILENAME)
    if _is_valid_gzip(repo_bpe) and not _looks_like_lfs_pointer(repo_bpe):
        return repo_bpe

    cache_bpe = os.path.join(os.path.expanduser('~'), '.cache', 'audioclip', BPE_FILENAME)
    if _is_valid_gzip(cache_bpe):
        return cache_bpe

    return _download_bpe(cache_bpe)


@lru_cache()
def default_bpe():
    return _resolve_bpe_path()


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
