// Copyright (c) 2026 LLMs For All, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

struct hot_policy {
    std::vector<int>     queue;
    std::array<int, 256> slots;

    explicit hot_policy(int capacity, unsigned seed, const std::vector<int> & initial = {}) {
        if (capacity < 8 || capacity > 256) {
            throw std::runtime_error("invalid hot capacity");
        }
        slots.fill(-1);
        if (!initial.empty()) {
            if (initial.size() != static_cast<size_t>(capacity)) {
                throw std::runtime_error("invalid initial expert count");
            }
            queue = initial;
            for (int i = 0; i < capacity; ++i) {
                const int id = queue[i];
                if (id < 0 || id >= 256 || slots[id] >= 0) {
                    throw std::runtime_error("invalid initial expert id");
                }
                slots[id] = i;
            }
            return;
        }
        std::vector<int> all(256);
        std::iota(all.begin(), all.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(all.begin(), all.end(), rng);
        queue.assign(all.begin(), all.begin() + capacity);
        for (int i = 0; i < capacity; ++i) {
            slots[queue[i]] = i;
        }
    }

    // Return (new logical expert, physical slot) for each cold admission.
    std::vector<std::pair<int, int>> update(const std::vector<int> & selected) {
        if (selected.size() != 8) {
            throw std::runtime_error("expected eight selected experts");
        }
        std::array<bool, 256> used{};
        for (int e : selected) {
            if (e < 0 || e >= 256 || used[e]) {
                throw std::runtime_error("invalid selected experts");
            }
            used[e] = true;
        }
        std::vector<int> next = selected;
        for (int e : queue) {
            if (!used[e]) {
                next.push_back(e);
            }
        }
        next.resize(queue.size());
        std::array<bool, 256> keep{};
        for (int e : next) {
            keep[e] = true;
        }
        std::vector<int> free;
        for (int e : queue) {
            if (!keep[e]) {
                free.push_back(slots[e]);
                slots[e] = -1;
            }
        }
        std::vector<std::pair<int, int>> loads;
        for (int e : selected) {
            if (slots[e] < 0) {
                if (free.empty()) {
                    throw std::runtime_error("no free hot slot");
                }
                int slot = free.back();
                free.pop_back();
                slots[e] = slot;
                loads.emplace_back(e, slot);
            }
        }
        queue = std::move(next);
        return loads;
    }
};
