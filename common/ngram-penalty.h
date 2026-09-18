#pragma once

// Escalating n-gram penalty against a fixed set of reference token sequences.
//
// Unlike the classic repetition penalty (which rescales logits and therefore
// behaves differently depending on logit scale and sign) this subtracts a
// fixed amount from the logit of any token that would extend a match against
// the reference set: a match of length n costs (n - start_n + 1) * scale,
// capped at max_penalty. Matches shorter than start_n cost nothing; with
// start_n == 1 every token that occurs in a reference costs at least scale. Only the
// reference sequences are consulted; the generation's own repeats are not.
//
// Intended use: after a detected pathology (a repeated tool call, a
// generation that looped to the cap) the client sends the offending text
// spans as references for the next request, so the model cannot re-emit
// them verbatim and has to do something else.

#include "llama.h"

#include <vector>

struct llama_sampler * common_ngram_penalty_init(
        const std::vector<std::vector<llama_token>> & refs,
        int32_t start_n,
        float   scale,
        float   max_penalty);
