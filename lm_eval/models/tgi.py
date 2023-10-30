""" TextSynth API
Implementation provided by Fabrice Bellard:
    https://github.com/EleutherAI/lm-evaluation-harness/issues/295

In order to use the API, you must have a valid TextSynth account and
enough credits.

Example usage:

    python main.py --model textsynth --model_args engine=gptj_6B --no_cache --tasks piqa

Homepage: https://textsynth.com/index.html
"""
import logging
import os
import json
import requests as _requests
import time
from tqdm import tqdm
from lm_eval.base import BaseLM
from lm_eval import utils
from transformers import LlamaTokenizer

logger = logging.getLogger(__name__)


def tgi_completion(**kwargs):
    """Query TGI API for completion.
    Retry with back-off until they respond.
    """
    backoff_time = 0.1
    while True:
        try:
            return _requests.post(**kwargs)
        except _requests.exceptions.RequestException:
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5


class TGILM(BaseLM):
    def __init__(self, model, truncate=False):
        super().__init__()

        self.model = model
        self.truncate = truncate
        self.api_url = "http://localhost:60/generate"
        self.tokenizer = LlamaTokenizer.from_pretrained(model, trust_remote_code=True)
        self.vocab_size = self.tokenizer.vocab_size


    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # NOTE: Turn on truncation to avoid errors on long inputs.
        return 4096

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        return "ignore"

    @property
    def device(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def compute_logprob(self, text):
        response = tgi_completion(
            url=self.api_url,
            json={
            "inputs": text,
            "parameters": {
                "decoder_input_details": True,
                "max_new_tokens": 1,
                "topk": 1,
                "do_sample": True,
                "details": True
            }
        })
        resp = response.json()
        try:
            prefills = resp["details"]["prefill"][1:]
        except:
            json_str = json.dumps(resp, indent=4)
            logger.error(
                f"The following response does not contain `logprobs`. Got:\n{json_str}"
            )
            assert False
        logprobs = [t["logprob"] for t in prefills]
        logprob = sum(logprobs)
        return logprob

    def compute_greedy(self, context, continuation):
        truth_tokens = self.tokenizer(continuation)['input_ids']
        response = tgi_completion(
            url=self.api_url,
            json={
            "inputs": context,
            "parameters": {
                "decoder_input_details": True,
                "max_new_tokens": len(truth_tokens),
                "topk": 50,
                "do_sample": True,
                "details": True
            }
        })
        resp = response.json()
        try:
            tokens = resp["details"]["tokens"]
        except:
            json_str = json.dumps(resp, indent=4)
            logger.error(
                f"The following response does not contain `logprobs`. Got:\n{json_str}"
            )
            assert False
        for i, j in zip(tokens, truth_tokens):
            if i["id"] != j:
                return False
        return True


    # def loglikelihood(self, requests):
    #     res = []
    #     for context, continuation in tqdm(requests):
    #         logprob = self.compute_logprob(context + continuation) - self.compute_logprob(context)
    #         is_greedy = self.compute_greedy(context, continuation)
    #         res.append((logprob, is_greedy))
    #     return res

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []

        def _collate(x):
            # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
            # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
            # we care about and so we need some kind of backup for when it isn't
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        for chunk in tqdm(
            list(utils.chunks(re_ord.get_reordered(), 20)),
            disable=disable_tqdm,
        ):
            inps = []
            ctxlens = []
            resps = []
            for cache_key, context_enc, continuation_enc in chunk:
                # max_length+1 because the API takes up to 2049 tokens, including the first context token
                inp = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                # TODO: the logic is much simpler if we just look at the length of continuation tokens
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length + 1)
                )

                inps.append(inp)
                ctxlens.append(ctxlen)
                text = self.tokenizer.decode(inp)
                response = tgi_completion(
                    url=self.api_url,
                    json={
                    "inputs": text,
                    "parameters": {
                        "decoder_input_details": True,
                        "max_new_tokens": 1,
                        "topk": 1,
                        "do_sample": True,
                        "details": True
                    }
                })
                resp = response.json()
                resps.append(resp)

            for resp, ctxlen, (cache_key, context_enc, continuation_enc) in zip(
                resps, ctxlens, chunk
            ):
                is_greedy = True
                try:
                    all_info = resp["details"]["prefill"][1:]
                    resp_tokens = resp["details"]["tokens"]
                except:
                    json_str = json.dumps(resp, indent=4)
                    logger.error(
                        f"The following response does not contain `logprobs`. Got:\n{json_str}"
                    )
                    assert False
                logprobs = [t["logprob"] for t in all_info]
                continuation_logprobs = sum(logprobs[ctxlen:])
                for i, j in zip(resp_tokens, continuation_enc):
                    if i["id"] != j:
                        is_greedy = False
                        break
    
                answer = (continuation_logprobs, is_greedy)
                res.append(answer)

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return re_ord.get_original(res)

    def greedy_until(self, requests):
        if not requests:
                return []
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        # todo: more intelligent batching for heterogeneous `until`
        for chunk, until in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), 20))
        ):
            inps = []
            resps = []
            for context, _ in chunk:
                context_enc = self.tok_encode(context)
                inp = context_enc[-(self.max_length - self.max_gen_toks) :]
                inps.append(inp)
                response = tgi_completion(
                    url=self.api_url,
                    json={
                    "inputs": context,
                    "parameters": {
                        "decoder_input_details": True,
                        "max_new_tokens": self.max_gen_toks,
                        "topk": 1,
                        "do_sample": True,
                        "details": True,
                        "stop": until["until"],
                    }
                })
                resp = response.json()
                resps.append(resp)

            for resp, (context, until_) in zip(resps, chunk):
                s = resp["generated_text"]

                for term in until_:
                    s = s.split(term)[0]

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until_), s)

                res.append(s)

        return re_ord.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
