use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::HashMap;

// ==============================
// Card utilities (0..=51)
// ==============================
#[pyfunction]
fn rank_of(c: u8) -> PyResult<i32> {
    if c > 51 {
        return Err(PyValueError::new_err("card out of range"));
    }
    Ok(2 + (c as i32 % 13))
}

#[pyfunction]
fn suit_of(c: u8) -> PyResult<i32> {
    if c > 51 {
        return Err(PyValueError::new_err("card out of range"));
    }
    Ok((c / 13) as i32)
}

#[pyfunction]
fn make_deck() -> Vec<u8> {
    (0u8..52u8).collect()
}

#[pyfunction]
fn action_type_id(kind: u8) -> PyResult<i32> {
    if kind <= 3 {
        Ok(kind as i32)
    } else {
        Err(PyValueError::new_err("invalid action kind"))
    }
}

#[pyfunction]
fn round_label_id(label: &str) -> i32 {
    match label {
        "Preflop" => 0,
        "Flop" => 1,
        "Turn" => 2,
        "River" => 3,
        _ => 3,
    }
}

// ==============================
// Types (PyO3 classes)
// ==============================
#[pyclass]
#[derive(Clone)]
struct Action {
    /// 0=FOLD, 1=CHECK, 2=CALL, 3=RAISE_TO
    #[pyo3(get, set)]
    kind: u8,
    /// Only used when kind==3 (RAISE_TO); otherwise None
    #[pyo3(get, set)]
    amount: Option<i32>,
}

#[pymethods]
impl Action {
    #[new]
    #[pyo3(signature = (kind, amount=None))]
    fn new(kind: u8, amount: Option<i32>) -> PyResult<Self> {
        if kind > 3 {
            return Err(PyValueError::new_err("invalid action kind"));
        }
        Ok(Self { kind, amount })
    }
}

#[pyclass]
#[derive(Clone)]
struct PlayerState {
    #[pyo3(get, set)]
    hole: Option<(u8, u8)>,
    #[pyo3(get, set)]
    stack: i32,
    #[pyo3(get, set)]
    bet: i32,
    #[pyo3(get, set)]
    cont: i32,
    /// "active" | "folded" | "allin"
    #[pyo3(get, set)]
    status: String,
    #[pyo3(get, set)]
    rho: i64,
}

#[pymethods]
impl PlayerState {
    #[new]
    #[pyo3(signature = (hole=None, stack=0, bet=0, cont=0, status=None, rho=None))]
    fn new(
        hole: Option<(u8, u8)>,
        stack: i32,
        bet: i32,
        cont: i32,
        status: Option<String>,
        rho: Option<i64>,
    ) -> Self {
        Self {
            hole,
            stack,
            bet,
            cont,
            status: status.unwrap_or_else(|| "active".into()),
            rho: rho.unwrap_or(-1_000_000_000),
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct GameState {
    #[pyo3(get, set)]
    button: usize,
    #[pyo3(get, set)]
    round_label: String,
    #[pyo3(get, set)]
    board: Vec<u8>,
    #[pyo3(get, set)]
    undealt: Vec<u8>,
    #[pyo3(get, set)]
    players: Vec<PlayerState>,
    #[pyo3(get, set)]
    current_bet: i32,
    #[pyo3(get, set)]
    min_raise: i32,
    #[pyo3(get, set)]
    tau: i64,
    #[pyo3(get, set)]
    next_to_act: Option<usize>,
    #[pyo3(get, set)]
    step_idx: i64,
    #[pyo3(get, set)]
    pot: i32,
    #[pyo3(get, set)]
    sb: i32,
    #[pyo3(get, set)]
    bb: i32,
    /// (player_id, action_id, amount, round_id)
    #[pyo3(get, set)]
    actions_log: Vec<(usize, i32, i32, i32)>,
}

#[pymethods]
impl GameState {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        button,
        round_label,
        board,
        undealt,
        players,
        current_bet,
        min_raise,
        tau,
        next_to_act=None,
        step_idx=0,
        pot=0,
        sb=1,
        bb=2
    ))]
    fn new(
        button: usize,
        round_label: String,
        board: Vec<u8>,
        undealt: Vec<u8>,
        players: Vec<PlayerState>,
        current_bet: i32,
        min_raise: i32,
        tau: i64,
        next_to_act: Option<usize>,
        step_idx: i64,
        pot: i32,
        sb: i32,
        bb: i32,
    ) -> Self {
        Self {
            button,
            round_label,
            board,
            undealt,
            players,
            current_bet,
            min_raise,
            tau,
            next_to_act,
            step_idx,
            pot,
            sb,
            bb,
            actions_log: Vec::new(),
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct LegalActionInfo {
    #[pyo3(get, set)]
    actions: Vec<Action>,
    #[pyo3(get, set)]
    min_raise_to: Option<i32>,
    #[pyo3(get, set)]
    max_raise_to: Option<i32>,
    #[pyo3(get, set)]
    has_raise_right: Option<bool>,
}

#[pymethods]
impl LegalActionInfo {
    #[new]
    #[pyo3(signature = (actions, min_raise_to=None, max_raise_to=None, has_raise_right=None))]
    fn new(
        actions: Vec<Action>,
        min_raise_to: Option<i32>,
        max_raise_to: Option<i32>,
        has_raise_right: Option<bool>,
    ) -> Self {
        Self {
            actions,
            min_raise_to,
            max_raise_to,
            has_raise_right,
        }
    }
}

// ==============================
// Diff structs (small payloads)
// ==============================
#[pyclass]
#[derive(Clone)]
struct PlayerDiff {
    #[pyo3(get)]
    idx: usize,
    #[pyo3(get)]
    stack: i32,
    #[pyo3(get)]
    bet: i32,
    #[pyo3(get)]
    cont: i32,
    #[pyo3(get)]
    rho: i64,
    #[pyo3(get)]
    status: Option<String>, // None = unchanged
}

#[pymethods]
impl PlayerDiff {
    #[new]
    #[pyo3(signature = (idx, stack, bet, cont, rho, status=None))]
    fn new(
        idx: usize,
        stack: i32,
        bet: i32,
        cont: i32,
        rho: i64,
        status: Option<String>,
    ) -> Self {
        Self {
            idx,
            stack,
            bet,
            cont,
            rho,
            status,
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct StepDiff {
    #[pyo3(get)]
    next_to_act: Option<usize>,
    #[pyo3(get)]
    step_idx: i64,
    #[pyo3(get)]
    current_bet: i32,
    #[pyo3(get)]
    min_raise: i32,
    #[pyo3(get)]
    tau: i64,
    #[pyo3(get)]
    pot: i32,
    #[pyo3(get)]
    round_label: Option<String>, // None = unchanged
    #[pyo3(get)]
    board_drawn: Vec<u8>,        // newly dealt cards this transition
    #[pyo3(get)]
    actions_log_push: Option<(usize, i32, i32, i32)>,
    #[pyo3(get)]
    player_updates: Vec<PlayerDiff>,
}

#[pymethods]
impl StepDiff {
    #[new]
    #[pyo3(signature = (
        next_to_act=None, step_idx=0, current_bet=0, min_raise=0, tau=0, pot=0,
        round_label=None, board_drawn=vec![], actions_log_push=None, player_updates=vec![]
    ))]
    fn new(
        next_to_act: Option<usize>,
        step_idx: i64,
        current_bet: i32,
        min_raise: i32,
        tau: i64,
        pot: i32,
        round_label: Option<String>,
        board_drawn: Vec<u8>,
        actions_log_push: Option<(usize, i32, i32, i32)>,
        player_updates: Vec<PlayerDiff>,
    ) -> Self {
        Self {
            next_to_act,
            step_idx,
            current_bet,
            min_raise,
            tau,
            pot,
            round_label,
            board_drawn,
            actions_log_push,
            player_updates,
        }
    }
}

// ==============================
// Fast 7-card hand evaluation
// ==============================

/// Hand categories (match your Python constants)
#[repr(i32)]
#[derive(Copy, Clone, Eq, PartialEq)]
enum HandCat {
    High = 0,
    OnePair = 1,
    TwoPair = 2,
    Trips = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    Quads = 7,
    StraightFlush = 8,
}

#[inline(always)]
fn rank_idx(card: i32) -> usize { (card % 13) as usize } // 0..12 = 2..A
#[inline(always)]
fn suit_idx(card: i32) -> usize { (card / 13) as usize } // 0..3

// Find straight high card from a 13-bit rank mask (bits 0..12 represent ranks 2..A)
#[inline(always)]
fn straight_high(mask: u16) -> Option<i32> {
    // Wheel: A-2-3-4-5 (bits {0..3} and bit 12)
    if (mask & 0b0001_1111) == 0b0001_1111 && (mask & (1 << 12)) != 0 {
        return Some(5);
    }
    // Any 5 consecutive ranks (2..6 up to 10..A)
    let m = mask;
    // check windows of size 5: [0..4], [1..5], ... [8..12]
    for hi in (4..=12).rev() {
        let window = ((m >> (hi - 4)) & 0b1_1111) as u16;
        if window == 0b1_1111 {
            return Some((hi as i32) + 2); // hi index 0=2, so +2
        }
    }
    None
}

#[inline(always)]
fn top_k_from_mask(mask: u16, k: usize, out: &mut [i32; 5]) {
    let mut idx = 0;
    for r in (0..13).rev() {
        if (mask & (1 << r)) != 0 {
            out[idx] = (r as i32) + 2;
            idx += 1;
            if idx == k { break; }
        }
    }
    // zero-pad if needed
    while idx < k {
        out[idx] = 0;
        idx += 1;
    }
}

/// Internal Rust-only version that returns Vec<i32> for tiebreakers
fn best5_rank_from_7_rust(cards7: &[i32; 7]) -> (i32, Vec<i32>) {
    // 4 suit masks, 13 rank counts
    let mut suit_masks: [u16; 4] = [0; 4];
    let mut rc: [u8; 13] = [0; 13];

    for &c in cards7 {
        let r = rank_idx(c);
        let s = suit_idx(c);
        suit_masks[s] |= 1 << r;
        rc[r] += 1;
    }

    // overall rank mask
    let mut all_mask: u16 = 0;
    for r in 0..13 {
        if rc[r] > 0 { all_mask |= 1 << r; }
    }

    // Flush / Straight Flush?
    let mut flush_suit: Option<usize> = None;
    for s in 0..4 {
        if suit_masks[s].count_ones() >= 5 {
            flush_suit = Some(s);
            break;
        }
    }

    // Straight Flush?
    if let Some(s) = flush_suit {
        if let Some(sh) = straight_high(suit_masks[s]) {
            return (HandCat::StraightFlush as i32, vec![sh]);
        }
    }

    // Rank multiplicities, collect (count, rank) pairs on stack
    let mut four: Option<i32> = None;
    let mut trips: [i32; 2] = [0; 2]; let mut n_trips = 0;
    let mut pairs: [i32; 3] = [0; 3]; let mut n_pairs = 0;
    let mut highs: [i32; 5] = [0; 5]; let mut n_highs = 0;

    for r in (0..13).rev() {
        match rc[r] {
            4 => { four = Some((r as i32) + 2); }
            3 => {
                if n_trips < 2 { trips[n_trips] = (r as i32) + 2; n_trips += 1; }
            }
            2 => {
                if n_pairs < 3 { pairs[n_pairs] = (r as i32) + 2; n_pairs += 1; }
            }
            1 => {
                if n_highs < 5 { highs[n_highs] = (r as i32) + 2; n_highs += 1; }
            }
            _ => {}
        }
    }

    // Quads?
    if let Some(q) = four {
        let mut kicker = 0;
        for r in (0..13).rev() {
            let rv = (r as i32) + 2;
            if rv != q && rc[r] > 0 { kicker = rv; break; }
        }
        return (HandCat::Quads as i32, vec![q, kicker]);
    }

    // Full House?
    if n_trips >= 1 && (n_pairs >= 1 || n_trips >= 2) {
        let t = trips[0];
        let p = if n_pairs >= 1 { pairs[0] } else { trips[1] };
        return (HandCat::FullHouse as i32, vec![t, p]);
    }

    // Flush?
    if let Some(s) = flush_suit {
        let mut top5 = [0; 5];
        top_k_from_mask(suit_masks[s], 5, &mut top5);
        return (HandCat::Flush as i32, top5.to_vec());
    }

    // Straight?
    if let Some(sh) = straight_high(all_mask) {
        return (HandCat::Straight as i32, vec![sh]);
    }

    // Trips?
    if n_trips >= 1 {
        let t = trips[0];
        let mut kick = [0; 2]; let mut k = 0;
        for r in (0..13).rev() {
            let rv = (r as i32) + 2;
            if rc[r] > 0 && rv != t {
                kick[k] = rv; k += 1;
                if k == 2 { break; }
            }
        }
        return (HandCat::Trips as i32, vec![t, kick[0], kick[1]]);
    }

    // Two Pair?
    if n_pairs >= 2 {
        let hp = pairs[0]; let lp = pairs[1];
        let mut kicker = 0;
        for r in (0..13).rev() {
            let rv = (r as i32) + 2;
            if rc[r] > 0 && rv != hp && rv != lp { kicker = rv; break; }
        }
        return (HandCat::TwoPair as i32, vec![hp, lp, kicker]);
    }

    // One Pair?
    if n_pairs == 1 {
        let p = pairs[0];
        let mut k = [0; 3]; let mut i = 0;
        for r in (0..13).rev() {
            let rv = (r as i32) + 2;
            if rc[r] > 0 && rv != p {
                k[i] = rv; i += 1;
                if i == 3 { break; }
            }
        }
        return (HandCat::OnePair as i32, vec![p, k[0], k[1], k[2]]);
    }

    // High card: top 5 ranks
    let mut top5 = [0; 5];
    top_k_from_mask(all_mask, 5, &mut top5);
    (HandCat::High as i32, top5.to_vec())
}

/// Return (category, tiebreakers as PyTuple) – tiebreak semantics match your Python.
/// cards: exactly 7 ints in 0..51
#[pyfunction]
fn best5_rank_from_7_py<'py>(py: Python<'py>, cards: Vec<i32>) -> PyResult<(i32, Bound<'py, PyTuple>)> {
    assert!(cards.len() == 7, "need 7 cards");
    // 4 suit masks, 13 rank counts
    let mut suit_masks: [u16; 4] = [0; 4];
    let mut rc: [u8; 13] = [0; 13];

    for &c in &cards {
        let r = rank_idx(c);
        let s = suit_idx(c);
        suit_masks[s] |= 1 << r;
        rc[r] += 1;
    }

    // overall rank mask
    let mut all_mask: u16 = 0;
    for r in 0..13 {
        if rc[r] > 0 { all_mask |= 1 << r; }
    }

    // Flush / Straight Flush?
    let mut flush_suit: Option<usize> = None;
    for s in 0..4 {
        if suit_masks[s].count_ones() >= 5 {
            flush_suit = Some(s);
            break;
        }
    }

    // Straight Flush?
    if let Some(s) = flush_suit {
        if let Some(sh) = straight_high(suit_masks[s]) {
            // Straight Flush: (sh,)
            let tb = PyTuple::new_bound(py, [sh]);
            return Ok((HandCat::StraightFlush as i32, tb));
        }
    }

    // Rank multiplicities, collect (count, rank) pairs on stack
    // We'll select patterns: 4,3,2,1 without sorting big arrays.
    let mut four: Option<i32> = None;
    let mut trips: [i32; 2] = [0; 2]; let mut n_trips = 0;
    let mut pairs: [i32; 3] = [0; 3]; let mut n_pairs = 0;
    let mut highs: [i32; 5] = [0; 5]; let mut n_highs = 0;

    for r in (0..13).rev() {
        match rc[r] {
            4 => { four = Some((r as i32) + 2); }
            3 => {
                if n_trips < 2 { trips[n_trips] = (r as i32) + 2; n_trips += 1; }
            }
            2 => {
                if n_pairs < 3 { pairs[n_pairs] = (r as i32) + 2; n_pairs += 1; }
            }
            1 => {
                if n_highs < 5 { highs[n_highs] = (r as i32) + 2; n_highs += 1; }
            }
            _ => {}
        }
    }

    // Quads?
    if let Some(q) = four {
        // kicker = highest not equal to q
        let mut kicker = 0;
        for r in (0..13).rev() {
            let rv = (r as i32) + 2;
            if rv != q && rc[r] > 0 { kicker = rv; break; }
        }
        let tb = PyTuple::new_bound(py, [q, kicker]);
        return Ok((HandCat::Quads as i32, tb));
    }

    // Full House?
    if n_trips >= 1 && (n_pairs >= 1 || n_trips >= 2) {
        let t = trips[0];
        let p = if n_pairs >= 1 { pairs[0] } else { trips[1] };
        let tb = PyTuple::new_bound(py, [t, p]);
        return Ok((HandCat::FullHouse as i32, tb));
    }

    // Flush?
    if let Some(s) = flush_suit {
        let mut top5 = [0; 5];
        top_k_from_mask(suit_masks[s], 5, &mut top5);
        let tb = PyTuple::new_bound(py, top5);
        return Ok((HandCat::Flush as i32, tb));
    }

    // Straight?
    if let Some(sh) = straight_high(all_mask) {
        let tb = PyTuple::new_bound(py, [sh]);
        return Ok((HandCat::Straight as i32, tb));
    }

    // Trips?
    if n_trips >= 1 {
        // trips + top 2 kickers
        let t = trips[0];
        let mut kick = [0; 2]; let mut k = 0;
        for r in (0..13).rev() {
            let rv = (r as i32) + 2;
            if rc[r] > 0 && rv != t {
                kick[k] = rv; k += 1;
                if k == 2 { break; }
            }
        }
        let tb = PyTuple::new_bound(py, [t, kick[0], kick[1]]);
        return Ok((HandCat::Trips as i32, tb));
    }

    // Two Pair?
    if n_pairs >= 2 {
        let hp = pairs[0]; let lp = pairs[1];
        let mut kicker = 0;
        for r in (0..13).rev() {
            let rv = (r as i32) + 2;
            if rc[r] > 0 && rv != hp && rv != lp { kicker = rv; break; }
        }
        let tb = PyTuple::new_bound(py, [hp, lp, kicker]);
        return Ok((HandCat::TwoPair as i32, tb));
    }

    // One Pair?
    if n_pairs == 1 {
        let p = pairs[0];
        let mut k = [0; 3]; let mut i = 0;
        for r in (0..13).rev() {
            let rv = (r as i32) + 2;
            if rc[r] > 0 && rv != p {
                k[i] = rv; i += 1;
                if i == 3 { break; }
            }
        }
        let tb = PyTuple::new_bound(py, [p, k[0], k[1], k[2]]);
        return Ok((HandCat::OnePair as i32, tb));
    }

    // High card: top 5 ranks
    let mut top5 = [0; 5];
    top_k_from_mask(all_mask, 5, &mut top5);
    let tb = PyTuple::new_bound(py, top5);
    Ok((HandCat::High as i32, tb))
}

/// (Optional) Batch API to amortize FFI
#[pyfunction]
fn best5_rank_from_7_batch_py<'py>(py: Python<'py>, hands: Vec<[i32; 7]>) -> PyResult<(Vec<i32>, Vec<Bound<'py, PyTuple>>)> {
    let mut cats = Vec::with_capacity(hands.len());
    let mut tbs  = Vec::with_capacity(hands.len());
    for h in hands {
        let (c, tb) = best5_rank_from_7_py(py, h.to_vec())?;
        cats.push(c);
        tbs.push(tb);
    }
    Ok((cats, tbs))
}

// ==============================
// Pure helpers (no &self) to avoid borrow conflicts
// ==============================
fn one_survivor(s: &GameState) -> Option<usize> {
    let alive: Vec<usize> = s
        .players
        .iter()
        .enumerate()
        .filter(|(_, p)| p.status != "folded")
        .map(|(i, _)| i)
        .collect();
    if alive.len() == 1 {
        Some(alive[0])
    } else {
        None
    }
}
fn everyone_allin_or_folded(s: &GameState) -> bool {
    s.players
        .iter()
        .all(|p| p.status != "active" || p.stack == 0)
}
fn round_open(s: &GameState) -> bool {
    for p in s.players.iter() {
        if p.status == "active" {
            let owe = (s.current_bet - p.bet).max(0);
            if p.rho < s.tau || owe > 0 {
                return true;
            }
        }
    }
    false
}
fn deal_next_street(s: &mut GameState) -> PyResult<()> {
    match s.round_label.as_str() {
        "Preflop" => {
            for _ in 0..3 {
                let c = s
                    .undealt
                    .pop()
                    .ok_or_else(|| PyValueError::new_err("deck underflow"))?;
                s.board.push(c);
            }
            s.round_label = "Flop".into();
        }
        "Flop" => {
            let c = s
                .undealt
                .pop()
                .ok_or_else(|| PyValueError::new_err("deck underflow"))?;
            s.board.push(c);
            s.round_label = "Turn".into();
        }
        "Turn" => {
            let c = s
                .undealt
                .pop()
                .ok_or_else(|| PyValueError::new_err("deck underflow"))?;
            s.board.push(c);
            s.round_label = "River".into();
        }
        _ => return Err(PyValueError::new_err("No further streets to deal")),
    }
    Ok(())
}
fn reset_round(n: usize, s: &mut GameState) {
    for p in &mut s.players {
        p.bet = 0;
        if p.status == "active" {
            p.rho = -1_000_000_000;
        }
    }
    s.current_bet = 0;
    s.min_raise = s.bb;
    s.tau = 0;
    let mut x = (s.button + 1) % n;
    for _ in 0..n {
        if s.players[x].status == "active" {
            s.next_to_act = Some(x);
            return;
        }
        x = (x + 1) % n;
    }
    s.next_to_act = None;
}
fn settle_showdown(n: usize, s: &GameState) -> PyResult<Vec<i32>> {
    let a: Vec<usize> = s
        .players
        .iter()
        .enumerate()
        .filter(|(_, p)| p.status != "folded")
        .map(|(i, _)| i)
        .collect();
    let mut levels: Vec<i32> = s.players.iter().map(|p| p.cont).filter(|v| *v > 0).collect();
    levels.sort();
    levels.dedup();
    if levels.is_empty() {
        return Ok(vec![0; n]);
    }

    let mut ranks: HashMap<usize, (i32, Vec<i32>)> = HashMap::new();
    for i in &a {
        let hole = s.players[*i]
            .hole
            .ok_or_else(|| PyValueError::new_err("missing hole cards"))?;
        let mut seven = [0i32; 7];
        seven[0] = hole.0 as i32;
        seven[1] = hole.1 as i32;
        for (k, c) in s.board.iter().enumerate() {
            seven[2 + k] = *c as i32;
        }
        let (cat, tb) = best5_rank_from_7_rust(&seven);
        ranks.insert(*i, (cat, tb));
    }

    let mut rewards = vec![0i32; n];
    let mut y_prev = 0i32;
    let mut carry = 0i32;
    let mut last_nonempty_winners: Option<Vec<usize>> = None;

    for y in levels {
        let contributors_count = s.players.iter().filter(|p| p.cont >= y).count() as i32;
        let pk = contributors_count * (y - y_prev) + carry;
        let elig: Vec<usize> = a.iter().cloned().filter(|i| s.players[*i].cont >= y).collect();

        if !elig.is_empty() {
            let best_val = elig
                .iter()
                .map(|i| ranks.get(i).unwrap())
                .max_by(|x, y| match x.0.cmp(&y.0) {
                    Ordering::Equal => x.1.cmp(&y.1),
                    o => o,
                })
                .unwrap()
                .clone();

            let winners: Vec<usize> = elig.into_iter().filter(|i| ranks[i] == best_val).collect();
            last_nonempty_winners = Some(winners.clone());
            let share = pk / winners.len() as i32;
            let rem = pk % winners.len() as i32;
            for w in &winners {
                rewards[*w] += share;
            }
            if rem > 0 {
                let start = (s.button + 1) % n;
                let mut ordered = winners.clone();
                ordered.sort_by_key(|j| ((*j + n) - start) % n);
                let len = ordered.len();
                for k in 0..rem as usize {
                    rewards[ordered[k % len]] += 1;
                }
            }
            carry = 0;
        } else {
            carry = pk;
        }
        y_prev = y;
    }

    if carry > 0 {
        if let Some(winners) = last_nonempty_winners {
            let share = carry / winners.len() as i32;
            let rem = carry % winners.len() as i32;
            for w in &winners {
                rewards[*w] += share;
            }
            if rem > 0 {
                let start = (s.button + 1) % n;
                let mut ordered = winners.clone();
                ordered.sort_by_key(|j| ((*j + n) - start) % n);
                let len = ordered.len();
                for k in 0..rem as usize {
                    rewards[ordered[k % len]] += 1;
                }
            }
        }
    }

    // Convert to net winnings
    let mut rl: Vec<i32> = Vec::with_capacity(n);
    for i in 0..n {
        rl.push(rewards[i] - s.players[i].cont);
    }
    Ok(rl)
}
fn legal_actions_from(s: &GameState) -> PyResult<LegalActionInfo> {
    let i = match s.next_to_act {
        Some(x) => x,
        None => return Ok(LegalActionInfo::new(vec![], None, None, None)),
    };
    let p = &s.players[i];
    if p.status != "active" {
        return Ok(LegalActionInfo::new(vec![], None, None, None));
    }

    let owe = (s.current_bet - p.bet).max(0);
    let mut acts = Vec::<Action>::new();
    if owe > 0 {
        acts.push(Action { kind: 0, amount: None }); // FOLD
    }
    if owe == 0 {
        acts.push(Action { kind: 1, amount: None }); // CHECK
    }
    if owe > 0 {
        acts.push(Action { kind: 2, amount: None }); // CALL
    }

    let can_raise = (p.status == "active") && (p.stack > 0);
    if !can_raise {
        return Ok(LegalActionInfo::new(acts, None, None, None));
    }

    let min_to = if s.current_bet == 0 {
        s.min_raise.max(1)
    } else {
        s.current_bet + s.min_raise
    };
    let max_to = p.bet + p.stack;
    let has_rr = (p.rho < s.tau) || (s.current_bet == 0);
    if max_to > s.current_bet {
        acts.push(Action { kind: 3, amount: None }); // RAISE_TO
        return Ok(LegalActionInfo::new(
            acts,
            Some(min_to),
            Some(max_to),
            Some(has_rr),
        ));
    }
    Ok(LegalActionInfo::new(acts, None, None, None))
}
fn advance_round_if_needed_internal(n: usize, s: &mut GameState) -> PyResult<(bool, Option<Vec<i32>>)> {
    if let Some(lone) = one_survivor(s) {
        let mut rewards = vec![0i32; n];
        for i in 0..n {
            rewards[i] = if i == lone { s.pot } else { 0 } - s.players[i].cont;
        }
        return Ok((true, Some(rewards)));
    }

    if !round_open(s) {
        if (s.round_label == "Preflop" || s.round_label == "Flop" || s.round_label == "Turn")
            && everyone_allin_or_folded(s)
        {
            while s.round_label != "River" {
                deal_next_street(s)?;
            }
            s.round_label = "Showdown".into();
            let rewards = settle_showdown(n, s)?;
            return Ok((true, Some(rewards)));
        }

        match s.round_label.as_str() {
            "Preflop" => {
                deal_next_street(s)?;
                reset_round(n, s);
                return Ok((false, None));
            }
            "Flop" => {
                deal_next_street(s)?;
                reset_round(n, s);
                return Ok((false, None));
            }
            "Turn" => {
                deal_next_street(s)?;
                reset_round(n, s);
                return Ok((false, None));
            }
            "River" => {
                s.round_label = "Showdown".into();
                let rewards = settle_showdown(n, s)?;
                return Ok((true, Some(rewards)));
            }
            _ => {}
        }
    }
    Ok((false, None))
}

// ==============================
// Engine with internal state + diffs
// ==============================
#[pyclass]
struct NLHEngine {
    n: usize,
    sb: i32,
    bb: i32,
    start_stack: i32,
    rng: StdRng,
    cur: Option<GameState>,
}

#[pymethods]
impl NLHEngine {
    #[new]
    #[pyo3(signature = (sb=1, bb=2, start_stack=100, num_players=6, seed=None))]
    fn new(
        sb: i32,
        bb: i32,
        start_stack: i32,
        num_players: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if num_players != 6 {
            return Err(PyValueError::new_err(
                "Engine fixed to 6 players per spec",
            ));
        }
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Ok(Self {
            n: num_players,
            sb,
            bb,
            start_stack,
            rng,
            cur: None,
        })
    }

    #[getter]
    fn N(&self) -> usize {
        self.n
    }

    /// Reset internal state and return a one-time full snapshot (Python builds its mirror from this)
    fn reset_hand(&mut self, button: usize) -> PyResult<GameState> {
        let s = self.reset_hand_internal(button)?;
        self.cur = Some(s.clone());
        Ok(s)
    }

    /// Reset internal state and update the provided Python GameState mirror in-place.
    /// Returns None (mirror already updated).
    fn reset_hand_apply_py<'py>(
        &mut self,
        py: Python<'py>,
        py_state: &Bound<'py, pyo3::PyAny>,
        button: usize,
    ) -> PyResult<()> {
        // build new internal rust state
        let s_new = self.reset_hand_internal(button)?;
        self.cur = Some(s_new.clone());
        let s2 = self.cur.as_ref().unwrap();

        // set scalars
        py_state.setattr("button", s2.button)?;
        py_state.setattr("round_label", s2.round_label.clone())?;
        py_state.setattr("current_bet", s2.current_bet)?;
        py_state.setattr("min_raise", s2.min_raise)?;
        py_state.setattr("tau", s2.tau)?;
        py_state.setattr("next_to_act", s2.next_to_act)?;
        py_state.setattr("step_idx", s2.step_idx)?;
        py_state.setattr("pot", s2.pot)?;
        py_state.setattr("sb", s2.sb)?;
        py_state.setattr("bb", s2.bb)?;

        // board: clear
        let board_obj = py_state.getattr("board")?;
        let board_py = board_obj.downcast::<PyList>()?;
        let empty_list = pyo3::types::PyList::empty_bound(py);
        board_py.set_slice(0, board_py.len(), &empty_list)?;

        // actions_log: clear
        let al_obj = py_state.getattr("actions_log")?;
        let al_py = al_obj.downcast::<PyList>()?;
        let empty_list2 = pyo3::types::PyList::empty_bound(py);
        al_py.set_slice(0, al_py.len(), &empty_list2)?;

        // players: update in-place; assume list length is fixed at N
        let players_obj = py_state.getattr("players")?;
        let players_py = players_obj.downcast::<PyList>()?;
        for (idx, p_new) in s2.players.iter().enumerate() {
            let p_obj = players_py.get_item(idx)?;
            p_obj.setattr("hole", p_new.hole)?;
            p_obj.setattr("stack", p_new.stack)?;
            p_obj.setattr("bet", p_new.bet)?;
            p_obj.setattr("cont", p_new.cont)?;
            p_obj.setattr("status", p_new.status.clone())?;
            p_obj.setattr("rho", p_new.rho)?;
        }

        Ok(())
    }

    /// Optional: export full snapshot (debug)
    fn export_snapshot(&self) -> PyResult<GameState> {
        match &self.cur {
            Some(s) => Ok(s.clone()),
            None => Err(PyValueError::new_err("no state")),
        }
    }

    /// Legal actions from current internal state
    fn legal_actions_now(&self) -> PyResult<LegalActionInfo> {
        let s = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        legal_actions_from(s)
    }

    /// Step using internal state; returns (done, rewards?, diff)
    fn step_diff(&mut self, a: &Action) -> PyResult<(bool, Option<Vec<i32>>, StepDiff)> {
        // Snapshot values BEFORE mutating (immutable short-lived borrow)
        let (board_len_before, round_before) = {
            let s = self
                .cur
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("no state"))?;
            (s.board.len(), s.round_label.clone())
        };

        // Now mutate
        let (done, rewards, last_log, _changed_mask) = {
            let s_mut = self
                .cur
                .as_mut()
                .ok_or_else(|| PyValueError::new_err("no state"))?;
            NLHEngine::step_on_internal(self.n, s_mut, a)?
        };

        // Build diff from AFTER state
        let s2 = self.cur.as_ref().unwrap();
        let mut board_drawn: Vec<u8> = vec![];
        if s2.board.len() > board_len_before {
            board_drawn.extend_from_slice(&s2.board[board_len_before..]);
        }
        let round_label_change = if s2.round_label != round_before {
            Some(s2.round_label.clone())
        } else {
            None
        };

        let mut player_updates = Vec::<PlayerDiff>::new();
        for (idx, p_new) in s2.players.iter().enumerate() {
            player_updates.push(PlayerDiff {
                idx,
                stack: p_new.stack,
                bet: p_new.bet,
                cont: p_new.cont,
                rho: p_new.rho,
                status: Some(p_new.status.clone()),
            });
        }

        let diff = StepDiff {
            next_to_act: s2.next_to_act,
            step_idx: s2.step_idx,
            current_bet: s2.current_bet,
            min_raise: s2.min_raise,
            tau: s2.tau,
            pot: s2.pot,
            round_label: round_label_change,
            board_drawn,
            actions_log_push: last_log,
            player_updates,
        };
        Ok((done, rewards, diff))
    }

    /// Advance round; returns (done, rewards?, diff)
    fn advance_round_if_needed_now(&mut self) -> PyResult<(bool, Option<Vec<i32>>, StepDiff)> {
        // Snapshot BEFORE
        let (board_len_before, round_before) = {
            let s = self
                .cur
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("no state"))?;
            (s.board.len(), s.round_label.clone())
        };

        // Mutate
        let (done, rewards) = {
            let s_mut = self
                .cur
                .as_mut()
                .ok_or_else(|| PyValueError::new_err("no state"))?;
            advance_round_if_needed_internal(self.n, s_mut)?
        };

        // AFTER
        let s2 = self.cur.as_ref().unwrap();
        let mut board_drawn: Vec<u8> = vec![];
        if s2.board.len() > board_len_before {
            board_drawn.extend_from_slice(&s2.board[board_len_before..]);
        }
        let round_label_change = if s2.round_label != round_before {
            Some(s2.round_label.clone())
        } else {
            None
        };

        let mut player_updates = Vec::<PlayerDiff>::new();
        for (idx, p_new) in s2.players.iter().enumerate() {
            player_updates.push(PlayerDiff {
                idx,
                stack: p_new.stack,
                bet: p_new.bet,
                cont: p_new.cont,
                rho: p_new.rho,
                status: Some(p_new.status.clone()),
            });
        }

        let last_log = s2.actions_log.last().cloned();

        let diff = StepDiff {
            next_to_act: s2.next_to_act,
            step_idx: s2.step_idx,
            current_bet: s2.current_bet,
            min_raise: s2.min_raise,
            tau: s2.tau,
            pot: s2.pot,
            round_label: round_label_change,
            board_drawn,
            actions_log_push: last_log,
            player_updates,
        };
        Ok((done, rewards, diff))
    }

    /// Same as step_apply_py but avoids creating/converting a Python Action object.
    /// kind: 0=FOLD,1=CHECK,2=CALL,3=RAISE_TO ; amount: None unless kind==3
    fn step_apply_py_raw<'py>(
        &mut self,
        py: Python<'py>,
        py_state: &Bound<'py, pyo3::PyAny>,
        kind: u8,
        amount: Option<i32>,
    ) -> PyResult<(bool, Option<Vec<i32>>)> {
        let a = Action { kind, amount };
        self.step_apply_py(py, py_state, &a)
    }

    /// Fast path: perform step on internal state AND update the given Python GameState mirror in-place.
    /// Returns (done, rewards?)
    fn step_apply_py<'py>(
        &mut self,
        py: Python<'py>,
        py_state: &Bound<'py, pyo3::PyAny>,
        a: &Action,
    ) -> PyResult<(bool, Option<Vec<i32>>)> {
        // --- SNAPSHOT BEFORE (immutable) ---
        let (board_len_before, round_before, _prev_players) = {
            let s = self.cur.as_ref().ok_or_else(|| PyValueError::new_err("no state"))?;
            let snap: Vec<(i32,i32,i32,i64,String)> = s.players.iter()
                .map(|p| (p.stack, p.bet, p.cont, p.rho, p.status.clone()))
                .collect();
            (s.board.len(), s.round_label.clone(), snap)
        };

        // --- MUTATE RUST STATE ---
        let (done, rewards, _last_log, changed_mask) = {
            let s_mut = self.cur.as_mut().ok_or_else(|| PyValueError::new_err("no state"))?;
            NLHEngine::step_on_internal(self.n, s_mut, a)?
        };

        // --- APPLY CHANGES TO PYTHON MIRROR IN ONE CALL ---
        let s2 = self.cur.as_ref().unwrap();

        // scalars
        py_state.setattr("step_idx", s2.step_idx)?;
        py_state.setattr("current_bet", s2.current_bet)?;
        py_state.setattr("min_raise", s2.min_raise)?;
        py_state.setattr("tau", s2.tau)?;
        py_state.setattr("pot", s2.pot)?;
        match s2.next_to_act {
            Some(v) => py_state.setattr("next_to_act", v)?,
            None => py_state.setattr("next_to_act", py.None())?,
        }
        if s2.round_label != round_before {
            py_state.setattr("round_label", s2.round_label.clone())?;
        }

        // board: append any newly dealt cards
        if s2.board.len() > board_len_before {
            let board_obj = py_state.getattr("board")?;
            let board_py = board_obj.downcast::<PyList>()?;
            for &c in &s2.board[board_len_before..] {
                board_py.append(c)?;
            }
        }

        // actions_log: push last entry (always one new)
        if let Some((i, aid, amt, rid)) = s2.actions_log.last().cloned() {
            let al_obj = py_state.getattr("actions_log")?;
            let al_py = al_obj.downcast::<PyList>()?;
            let tup = PyTuple::new_bound(py, &[i.into_py(py), aid.into_py(py), amt.into_py(py), rid.into_py(py)]);
            al_py.append(tup)?;
        }

        // players: update only changed ones using the bitmask
        let players_obj = py_state.getattr("players")?;
        let players_py = players_obj.downcast::<PyList>()?;
        let mut mask = changed_mask;
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            mask &= mask - 1; // clear lowest set bit
            let p_new = &s2.players[idx];
            let p_obj = players_py.get_item(idx)?;
            p_obj.setattr("stack", p_new.stack)?;
            p_obj.setattr("bet", p_new.bet)?;
            p_obj.setattr("cont", p_new.cont)?;
            p_obj.setattr("rho", p_new.rho)?;
            p_obj.setattr("status", p_new.status.clone())?;
        }

        Ok((done, rewards))
    }

    /// Fast path: advance round if needed, and update Python GameState mirror in-place.
    fn advance_round_if_needed_apply_py<'py>(
        &mut self,
        py: Python<'py>,
        py_state: &Bound<'py, pyo3::PyAny>,
    ) -> PyResult<(bool, Option<Vec<i32>>)> {
        // --- SNAPSHOT BEFORE ---
        let (board_len_before, round_before) = {
            let s = self.cur.as_ref().ok_or_else(|| PyValueError::new_err("no state"))?;
            (s.board.len(), s.round_label.clone())
        };

        // --- MUTATE RUST STATE ---
        let (done, rewards) = {
            let s_mut = self.cur.as_mut().ok_or_else(|| PyValueError::new_err("no state"))?;
            advance_round_if_needed_internal(self.n, s_mut)?
        };

        // --- APPLY TO PYTHON ---
        let s2 = self.cur.as_ref().unwrap();

        py_state.setattr("step_idx", s2.step_idx)?;
        py_state.setattr("current_bet", s2.current_bet)?;
        py_state.setattr("min_raise", s2.min_raise)?;
        py_state.setattr("tau", s2.tau)?;
        py_state.setattr("pot", s2.pot)?;
        match s2.next_to_act {
            Some(v) => py_state.setattr("next_to_act", v)?,
            None => py_state.setattr("next_to_act", py.None())?,
        }
        if s2.round_label != round_before {
            py_state.setattr("round_label", s2.round_label.clone())?;
        }

        if s2.board.len() > board_len_before {
            let board_obj = py_state.getattr("board")?;
            let board_py = board_obj.downcast::<PyList>()?;
            for &c in &s2.board[board_len_before..] {
                board_py.append(c)?;
            }
        }

        // push last log if any
        if let Some((i, aid, amt, rid)) = s2.actions_log.last().cloned() {
            let al_obj = py_state.getattr("actions_log")?;
            let al_py = al_obj.downcast::<PyList>()?;
            let tup = PyTuple::new_bound(py, &[i.into_py(py), aid.into_py(py), amt.into_py(py), rid.into_py(py)]);
            al_py.append(tup)?;
        }

        // players: update all (advance_round affects multiple players)
        let players_obj = py_state.getattr("players")?;
        let players_py = players_obj.downcast::<PyList>()?;
        for (idx, p_new) in s2.players.iter().enumerate() {
            let p_obj = players_py.get_item(idx)?;
            p_obj.setattr("stack", p_new.stack)?;
            p_obj.setattr("bet", p_new.bet)?;
            p_obj.setattr("cont", p_new.cont)?;
            p_obj.setattr("rho", p_new.rho)?;
            p_obj.setattr("status", p_new.status.clone())?;
        }

        Ok((done, rewards))
    }

    /// Fast legal-actions: return (mask, min_to, max_to, has_rr)
    /// mask bits: 1=FOLD, 2=CHECK, 4=CALL, 8=RAISE_TO
    fn legal_actions_bits_now(&self) -> PyResult<(u8, Option<i32>, Option<i32>, Option<bool>)> {
        let s = self.cur.as_ref().ok_or_else(|| PyValueError::new_err("no state"))?;
        let i = match s.next_to_act {
            Some(x) => x,
            None => return Ok((0, None, None, None)),
        };
        let p = &s.players[i];
        if p.status != "active" {
            return Ok((0, None, None, None));
        }

        let owe = (s.current_bet - p.bet).max(0);
        let mut mask: u8 = 0;
        if owe > 0 { mask |= 1; } // FOLD
        if owe == 0 { mask |= 2; } // CHECK
        if owe > 0 { mask |= 4; } // CALL

        let can_raise = (p.status == "active") && (p.stack > 0);
        if !can_raise {
            return Ok((mask, None, None, None));
        }

        let min_to = if s.current_bet == 0 {
            s.min_raise.max(1)
        } else {
            s.current_bet + s.min_raise
        };
        let max_to = p.bet + p.stack;
        let has_rr = (p.rho < s.tau) || (s.current_bet == 0);

        if max_to > s.current_bet {
            mask |= 8; // RAISE_TO
            Ok((mask, Some(min_to), Some(max_to), Some(has_rr)))
        } else {
            Ok((mask, None, None, None))
        }
    }
}

// ---- reset_hand for NLHEngine (mutates self.cur only through return) ----
impl NLHEngine {
    fn reset_hand_internal(&mut self, button: usize) -> PyResult<GameState> {
        let mut deck = make_deck();
        // Fisher–Yates
        for i in (1..deck.len()).rev() {
            let j = self.rng.gen_range(0..=i);
            deck.swap(i, j);
        }
        let mut players: Vec<PlayerState> = (0..self.n)
            .map(|_| PlayerState {
                hole: None,
                stack: self.start_stack,
                bet: 0,
                cont: 0,
                status: "active".into(),
                rho: -1_000_000_000,
            })
            .collect();

        for i in 0..self.n {
            let c1 = deck.pop().ok_or_else(|| PyValueError::new_err("deck underflow"))?;
            let c2 = deck.pop().ok_or_else(|| PyValueError::new_err("deck underflow"))?;
            players[i].hole = Some((c1, c2));
        }

        let board: Vec<u8> = Vec::new();
        let undealt = deck.clone();

        // blinds
        let sb_seat = (button + 1) % self.n;
        let bb_seat = (button + 2) % self.n;
        let sb_amt = self.sb.min(players[sb_seat].stack);
        players[sb_seat].stack -= sb_amt;
        players[sb_seat].bet += sb_amt;
        players[sb_seat].cont += sb_amt;
        if players[sb_seat].stack == 0 && sb_amt > 0 {
            players[sb_seat].status = "allin".into();
        }
        let bb_amt = self.bb.min(players[bb_seat].stack);
        players[bb_seat].stack -= bb_amt;
        players[bb_seat].bet += bb_amt;
        players[bb_seat].cont += bb_amt;
        if players[bb_seat].stack == 0 && bb_amt > 0 {
            players[bb_seat].status = "allin".into();
        }

        let current_bet = self.bb;
        let pot: i32 = players.iter().map(|p| p.cont).sum();

        Ok(GameState {
            button,
            round_label: "Preflop".into(),
            board,
            undealt,
            players,
            current_bet,
            min_raise: self.bb,
            tau: 0,
            next_to_act: Some((button + 3) % self.n),
            step_idx: 0,
            pot,
            sb: self.sb,
            bb: self.bb,
            actions_log: vec![],
        })
    }

    /// Static helper: perform a step on a provided mutable state (no &self borrow!)
    fn step_on_internal(
        n: usize,
        s: &mut GameState,
        a: &Action,
    ) -> PyResult<(bool, Option<Vec<i32>>, Option<(usize, i32, i32, i32)>, u64)> {
        let i = s
            .next_to_act
            .ok_or_else(|| PyValueError::new_err("no next_to_act"))?;
        if s.players[i].status != "active" {
            return Err(PyValueError::new_err("player not active"));
        }
        s.step_idx += 1;

        let mut changed: u64 = 0;

        let advance_next = |s: &mut GameState, i: usize, n: usize| {
            let mut j = (i + 1) % n;
            for _ in 0..n {
                let pj = &s.players[j];
                if pj.status == "active" {
                    let owej = (s.current_bet - pj.bet).max(0);
                    if pj.rho < s.tau || owej > 0 {
                        s.next_to_act = Some(j);
                        return;
                    }
                }
                j = (j + 1) % n;
            }
            s.next_to_act = None;
        };
        let add_chips = |s: &mut GameState, idx: usize, amount: i32| -> PyResult<()> {
            if amount < 0 {
                return Err(PyValueError::new_err("negative amount"));
            }
            if s.players[idx].stack < amount {
                return Err(PyValueError::new_err("insufficient stack"));
            }
            s.players[idx].stack -= amount;
            s.players[idx].bet += amount;
            s.players[idx].cont += amount;
            Ok(())
        };

        let owe = (s.current_bet - s.players[i].bet).max(0);
        let b_old = s.current_bet;

        match a.kind {
            0 => {
                // FOLD
                changed |= 1 << i;
                s.players[i].status = "folded".into();
                s.players[i].rho = s.step_idx;
                advance_next(s, i, n);
            }
            1 => {
                // CHECK
                if owe != 0 {
                    return Err(PyValueError::new_err("cannot CHECK when owing"));
                }
                changed |= 1 << i;
                s.players[i].rho = s.step_idx;
                advance_next(s, i, n);
            }
            2 => {
                // CALL
                if owe <= 0 {
                    return Err(PyValueError::new_err("cannot CALL when owe==0"));
                }
                changed |= 1 << i;
                let call_amt = owe.min(s.players[i].stack);
                add_chips(s, i, call_amt)?;
                if s.players[i].stack == 0 {
                    s.players[i].status = "allin".into();
                }
                s.players[i].rho = s.step_idx;
                advance_next(s, i, n);
            }
            3 => {
                // RAISE_TO
                changed |= 1 << i;
                let raise_to =
                    a.amount
                        .ok_or_else(|| PyValueError::new_err("RAISE_TO requires amount"))?;
                if raise_to <= s.current_bet {
                    return Err(PyValueError::new_err(
                        "raise_to must exceed current_bet",
                    ));
                }
                let max_to = s.players[i].bet + s.players[i].stack;
                if raise_to > max_to {
                    return Err(PyValueError::new_err("raise exceeds max_to"));
                }
                let has_rr = (s.players[i].rho < s.tau) || (s.current_bet == 0);
                let required_min_to = if s.current_bet == 0 {
                    s.min_raise.max(1)
                } else {
                    s.current_bet + s.min_raise
                };
                if !has_rr && raise_to != max_to {
                    return Err(PyValueError::new_err(
                        "only all-in allowed; raise rights are closed",
                    ));
                } else if has_rr && !(raise_to >= required_min_to || raise_to == max_to) {
                    return Err(PyValueError::new_err("raise below required minimum"));
                }
                let delta = raise_to - s.players[i].bet;
                add_chips(s, i, delta)?;
                if s.players[i].stack == 0 {
                    s.players[i].status = "allin".into();
                }
                s.current_bet = raise_to;
                s.players[i].rho = s.step_idx;
                let full_inc = raise_to - b_old;
                if full_inc >= s.min_raise {
                    s.tau = s.step_idx;
                    s.min_raise = full_inc;
                    for j in 0..n {
                        if j != i && s.players[j].status == "active" {
                            changed |= 1 << j;
                            s.players[j].rho = -1_000_000_000;
                        }
                    }
                }
                advance_next(s, i, n);
            }
            _ => return Err(PyValueError::new_err("unknown action type")),
        }

        // log
        let aid = a.kind as i32;
        let rid = round_label_id(&s.round_label);
        let mut log_amt = 0i32;
        if a.kind == 3 {
            log_amt = s.current_bet;
        }
        s.actions_log.push((i, aid, log_amt, rid));

        // recompute pot/current_bet
        s.pot = s.players.iter().map(|pl| pl.cont).sum();
        s.current_bet = s.players.iter().map(|pl| pl.bet).max().unwrap_or(0);

        let (done, rewards) = advance_round_if_needed_internal(n, s)?;
        let last_log = s.actions_log.last().cloned();
        Ok((done, rewards, last_log, changed))
    }
}

// ==============================
// Python module
// ==============================
#[pymodule]
fn nlhe_engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // classes
    m.add_class::<Action>()?;
    m.add_class::<PlayerState>()?;
    m.add_class::<GameState>()?;
    m.add_class::<LegalActionInfo>()?;
    m.add_class::<PlayerDiff>()?;
    m.add_class::<StepDiff>()?;
    m.add_class::<NLHEngine>()?;

    // functions
    m.add_function(wrap_pyfunction!(rank_of, m)?)?;
    m.add_function(wrap_pyfunction!(suit_of, m)?)?;
    m.add_function(wrap_pyfunction!(make_deck, m)?)?;
    m.add_function(wrap_pyfunction!(action_type_id, m)?)?;
    m.add_function(wrap_pyfunction!(round_label_id, m)?)?;
    m.add_function(wrap_pyfunction!(best5_rank_from_7_py, m)?)?;
    m.add_function(wrap_pyfunction!(best5_rank_from_7_batch_py, m)?)?;

    // convenience
    let __all__ = vec![
        "Action",
        "PlayerState",
        "GameState",
        "LegalActionInfo",
        "PlayerDiff",
        "StepDiff",
        "NLHEngine",
        "rank_of",
        "suit_of",
        "make_deck",
        "action_type_id",
        "round_label_id",
        "best5_rank_from_7_py",
        "best5_rank_from_7_batch_py",
    ];
    m.add("__all__", __all__)?;
    Ok(())
}
