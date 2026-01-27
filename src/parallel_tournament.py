"""
Speculative Parallel Pyramid Tournament for RESCue Verifier.

For N masks, the tournament works as follows:
- Possible winners: 0, 1, best(2-3), best(4-7), best(8-15), ...
- Each possible winner competes with the best from the next bracket

Example for N=32:
- Possible winners: [0, 1, best(2-3), best(4-7), best(8-15)]
- Challenger: best(16-31)
- All duels run in parallel, then resolve bracket using cached results.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed


def parallel_pyramid_tournament(verifier, image, masks, query, ranking):
    """
    SPECULATIVE PARALLEL Pyramid Tournament with bracket-based candidates.
    
    For N=8:
      Possible winners: index 0, index 1, best of [2-3]
      Challenger: best of [4-7]
      
    For N=32:
      Possible winners: 0, 1, best(2-3), best(4-7), best(8-15)
      Challenger: best(16-31)
    
    All possible duels computed in parallel, then resolved sequentially using cached results.
    """
    n = len(ranking)
    if n <= 1:
        return ranking
    
    # Build bracket boundaries: [1, 2, 4, 8, 16, ...]
    brackets = []
    step = 1
    while step < n:
        brackets.append(step)
        step *= 2
    
    if not brackets:
        return ranking
    
    # Get best mask from each bracket range
    # Bracket i covers indices [brackets[i-1], brackets[i]) for i >= 1
    # Bracket 0 is just index 0, Bracket 1 is just index 1
    def get_best_from_range(start, end):
        """Get mask_idx with highest score in ranking[start:end]"""
        best = None
        best_score = -1
        for i in range(start, min(end, n)):
            if ranking[i]['score'] > best_score:
                best_score = ranking[i]['score']
                best = ranking[i]
        return best
    
    # Build possible winners and challengers for each stage
    # Stage 0: 0 vs 1
    # Stage 1: winner vs best(2-3)
    # Stage 2: winner vs best(4-7)
    # etc.
    
    possible_winners = []  # List of ranking entries that could be winner before each challenge
    challengers = []  # List of (bracket_idx, ranking_entry) for each challenge
    
    # First two possible winners are always indices 0 and 1
    if n > 0:
        possible_winners.append(ranking[0])
    if n > 1:
        possible_winners.append(ranking[1])
        challengers.append(ranking[1])  # First challenge is 0 vs 1
    
    # For each subsequent bracket, add best from that range
    prev_boundary = 2
    for i, boundary in enumerate(brackets[2:], start=2):  # Skip first two (0, 1)
        if prev_boundary >= n:
            break
        next_boundary = boundary
        best_in_bracket = get_best_from_range(prev_boundary, next_boundary)
        if best_in_bracket:
            possible_winners.append(best_in_bracket)
            challengers.append(best_in_bracket)
        prev_boundary = next_boundary
    
    # Final challenger is best from remaining range
    if prev_boundary < n:
        final_challenger = get_best_from_range(prev_boundary, n)
        if final_challenger:
            challengers.append(final_challenger)
    
    if verifier.verbose:
        pw_indices = [r['mask_idx'] for r in possible_winners]
        ch_indices = [r['mask_idx'] for r in challengers]
        print(f"[Tournament] Possible winners: {pw_indices}")
        print(f"[Tournament] Challengers: {ch_indices}")
    
    # Build all possible duels:
    # For each challenger, duel with all possible winners accumulated so far
    all_duels = []  # List of (possible_winner_entry, challenger_entry)
    current_possible = [possible_winners[0]] if possible_winners else []
    
    for challenger in challengers:
        for pw in current_possible:
            if pw['mask_idx'] != challenger['mask_idx']:
                all_duels.append((pw, challenger))
        current_possible.append(challenger)
    
    if verifier.verbose:
        print(f"[Tournament] Parallel: {len(all_duels)} duels")
    
    if not all_duels:
        return ranking
    
    # Execute ALL duels in parallel
    duel_results = {}  # (winner_mask_idx, challenger_mask_idx) -> bool
    
    def run_duel(duel):
        pw_res, ch_res = duel
        pw_idx = pw_res['mask_idx']
        ch_idx = ch_res['mask_idx']
        result = verifier._compare_pair(image, masks[pw_idx], masks[ch_idx], pw_res, ch_res, query)
        return (pw_idx, ch_idx), result
    
    with ThreadPoolExecutor(max_workers=min(64, len(all_duels))) as executor:
        futures = [executor.submit(run_duel, d) for d in all_duels]
        for future in as_completed(futures):
            try:
                key, result = future.result()
                duel_results[key] = result
            except Exception as e:
                if verifier.verbose:
                    print(f"Duel error: {e}")
    
    # Resolve bracket using cached results
    current_winner = possible_winners[0] if possible_winners else None
    current_winner_rank_idx = 0
    
    for challenger in challengers:
        if current_winner is None:
            current_winner = challenger
            continue
            
        pw_idx = current_winner['mask_idx']
        ch_idx = challenger['mask_idx']
        
        # Look up precomputed result
        winner_wins = duel_results.get((pw_idx, ch_idx), True)
        
        if verifier.verbose:
            print(f"[Duel] {pw_idx} vs {ch_idx} -> {'Winner' if winner_wins else 'Challenger'}")
        
        if not winner_wins:
            current_winner = challenger
            # Find rank index
            for i, r in enumerate(ranking):
                if r['mask_idx'] == ch_idx:
                    current_winner_rank_idx = i
                    break
    
    # Move final winner to top
    if current_winner and current_winner_rank_idx != 0:
        winner_res = ranking.pop(current_winner_rank_idx)
        ranking.insert(0, winner_res)
        for i, r in enumerate(ranking):
            r['rank'] = i + 1
    
    if verifier.verbose:
        print(f"[Tournament] Winner: mask {current_winner['mask_idx'] if current_winner else 'none'}")
            
    return ranking
