use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Py;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const STATUS_ACTIVE: u8 = 0;
const STATUS_FOLDED: u8 = 1;
const STATUS_ALLIN: u8 = 2;

#[inline(always)]
fn status_to_str(s: u8) -> &'static str {
    match s {
        STATUS_FOLDED => "folded",
        STATUS_ALLIN => "allin",
        _ => "active",
    }
}

#[inline(always)]
fn str_to_status(v: &str) -> u8 {
    match v {
        "folded" => STATUS_FOLDED,
        "allin" => STATUS_ALLIN,
        _ => STATUS_ACTIVE,
    }
}

const ROUND_PREFLOP: u8 = 0;
const ROUND_FLOP: u8 = 1;
const ROUND_TURN: u8 = 2;
const ROUND_RIVER: u8 = 3;
const ROUND_SHOWDOWN: u8 = 4;

#[inline(always)]
fn round_to_str(r: u8) -> &'static str {
    match r {
        ROUND_FLOP => "Flop",
        ROUND_TURN => "Turn",
        ROUND_RIVER => "River",
        ROUND_SHOWDOWN => "Showdown",
        _ => "Preflop",
    }
}

#[inline(always)]
fn str_to_round(s: &str) -> u8 {
    match s {
        "Flop" => ROUND_FLOP,
        "Turn" => ROUND_TURN,
        "River" => ROUND_RIVER,
        "Showdown" => ROUND_SHOWDOWN,
        _ => ROUND_PREFLOP,
    }
}

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
    status: u8,
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
            status: status.map(|s| str_to_status(&s)).unwrap_or(STATUS_ACTIVE),
            rho: rho.unwrap_or(-1_000_000_000),
        }
    }

    #[getter]
    fn status(&self) -> &str {
        status_to_str(self.status)
    }

    #[setter]
    fn set_status(&mut self, v: &str) {
        self.status = str_to_status(v);
    }
}

#[pyclass]
#[derive(Clone)]
struct GameState {
    #[pyo3(get, set)]
    button: usize,
    round: u8,
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
            round: str_to_round(&round_label),
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

    #[getter]
    fn round_label(&self) -> &str {
        round_to_str(self.round)
    }

    #[setter]
    fn set_round_label(&mut self, v: &str) {
        self.round = str_to_round(v);
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
    fn new(idx: usize, stack: i32, bet: i32, cont: i32, rho: i64, status: Option<String>) -> Self {
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
    board_drawn: Vec<u8>, // newly dealt cards this transition
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
// Hand evaluation
// ==============================
#[derive(Copy, Clone, Eq, PartialEq)]
enum HandCategory {
    High = 0,
    OnePair = 1,
    TwoPair = 2,
    Trips = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    Four = 7,
    StraightFlush = 8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HandRank {
    cat: i32,
    tiebreak: [i32; 5],
}

impl HandRank {
    #[inline(always)]
    fn new(cat: i32, tiebreak: [i32; 5]) -> Self {
        Self { cat, tiebreak }
    }
}

impl Ord for HandRank {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.cat.cmp(&other.cat) {
            Ordering::Equal => {
                for (a, b) in self.tiebreak.iter().zip(other.tiebreak.iter()) {
                    match a.cmp(b) {
                        Ordering::Equal => continue,
                        o => return o,
                    }
                }
                Ordering::Equal
            }
            o => o,
        }
    }
}

impl PartialOrd for HandRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline(always)]
fn hand_rank_5(cards5: &[u8; 5]) -> HandRank {
    let mut ranks = [0i32; 5];
    let mut suits = [0u8; 5];
    for (i, &c) in cards5.iter().enumerate() {
        ranks[i] = 2 + (c as i32 % 13);
        suits[i] = c / 13;
    }
    ranks.sort_unstable_by(|a, b| b.cmp(a));
    let is_flush = suits.iter().all(|&s| s == suits[0]);

    let mut cnt = [0u8; 15];
    for &r in &ranks {
        cnt[r as usize] += 1;
    }

    let mut pairs = [0i32; 2];
    let mut pair_count = 0;
    let mut trips = 0i32;
    let mut has_trips = false;
    let mut quad = 0i32;
    let mut has_quad = false;
    for r in (2..=14).rev() {
        match cnt[r] {
            4 => {
                quad = r as i32;
                has_quad = true;
            }
            3 => {
                trips = r as i32;
                has_trips = true;
            }
            2 => {
                if pair_count < 2 {
                    pairs[pair_count] = r as i32;
                }
                pair_count += 1;
            }
            _ => {}
        }
    }

    let mut mask: u16 = 0;
    for &r in &ranks {
        mask |= 1 << r;
    }
    let mut s_high = 0i32;
    let mut has_straight = false;
    if (mask & ((1 << 14) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 2)))
        == ((1 << 14) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 2))
    {
        s_high = 5;
        has_straight = true;
    } else {
        for h in (5..=14).rev() {
            let pat = 0b1_1111 << (h - 4);
            if mask & pat == pat {
                s_high = h as i32;
                has_straight = true;
                break;
            }
        }
    }

    if is_flush && has_straight {
        return HandRank::new(HandCategory::StraightFlush as i32, [s_high, 0, 0, 0, 0]);
    }
    if has_quad {
        let kicker = ranks.iter().find(|&&r| r != quad).cloned().unwrap();
        return HandRank::new(HandCategory::Four as i32, [quad, kicker, 0, 0, 0]);
    }
    if has_trips && pair_count > 0 {
        return HandRank::new(HandCategory::FullHouse as i32, [trips, pairs[0], 0, 0, 0]);
    }
    if is_flush {
        return HandRank::new(HandCategory::Flush as i32, ranks);
    }
    if has_straight {
        return HandRank::new(HandCategory::Straight as i32, [s_high, 0, 0, 0, 0]);
    }
    if has_trips {
        let mut tb = [trips, 0, 0, 0, 0];
        let mut idx = 1;
        for &r in &ranks {
            if r != trips {
                tb[idx] = r;
                idx += 1;
            }
        }
        return HandRank::new(HandCategory::Trips as i32, tb);
    }
    if pair_count >= 2 {
        let p1 = pairs[0];
        let p2 = pairs[1];
        let kicker = ranks
            .iter()
            .find(|&&r| r != p1 && r != p2)
            .cloned()
            .unwrap();
        return HandRank::new(HandCategory::TwoPair as i32, [p1, p2, kicker, 0, 0]);
    }
    if pair_count == 1 {
        let pair = pairs[0];
        let mut tb = [pair, 0, 0, 0, 0];
        let mut idx = 1;
        for &r in &ranks {
            if r != pair {
                tb[idx] = r;
                idx += 1;
            }
        }
        return HandRank::new(HandCategory::OnePair as i32, tb);
    }
    HandRank::new(HandCategory::High as i32, ranks)
}

const COMB_IDX: [[usize; 5]; 21] = [
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 5],
    [0, 1, 2, 3, 6],
    [0, 1, 2, 4, 5],
    [0, 1, 2, 4, 6],
    [0, 1, 2, 5, 6],
    [0, 1, 3, 4, 5],
    [0, 1, 3, 4, 6],
    [0, 1, 3, 5, 6],
    [0, 1, 4, 5, 6],
    [0, 2, 3, 4, 5],
    [0, 2, 3, 4, 6],
    [0, 2, 3, 5, 6],
    [0, 2, 4, 5, 6],
    [0, 3, 4, 5, 6],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 6],
    [1, 2, 3, 5, 6],
    [1, 2, 4, 5, 6],
    [1, 3, 4, 5, 6],
    [2, 3, 4, 5, 6],
];

#[inline(always)]
fn best5_rank_from_7(cards7: &[u8; 7]) -> HandRank {
    let mut best = HandRank::new(-1, [0; 5]);
    for comb in COMB_IDX {
        let hand = [
            cards7[comb[0]],
            cards7[comb[1]],
            cards7[comb[2]],
            cards7[comb[3]],
            cards7[comb[4]],
        ];
        let r = hand_rank_5(&hand);
        if r > best {
            best = r;
        }
    }
    best
}

/// Python-friendly entry (parity with your eval.py)
#[pyfunction]
fn best5_rank_from_7_py(cards7: Vec<u8>) -> PyResult<(i32, Vec<i32>)> {
    if cards7.len() != 7 {
        return Err(PyValueError::new_err("cards7 must have length 7"));
    }
    let mut a = [0u8; 7];
    a.copy_from_slice(&cards7);
    let r = best5_rank_from_7(&a);
    Ok((r.cat, r.tiebreak.to_vec()))
}

// ==============================
// Pure helpers (no &self) to avoid borrow conflicts
// ==============================
#[inline(always)]
fn one_survivor(s: &GameState) -> Option<usize> {
    let mut survivor = None;
    for (i, p) in s.players.iter().enumerate() {
        if p.status != STATUS_FOLDED {
            if survivor.is_some() {
                return None;
            }
            survivor = Some(i);
        }
    }
    survivor
}
#[inline(always)]
fn everyone_allin_or_folded(s: &GameState) -> bool {
    for p in &s.players {
        if p.status == STATUS_ACTIVE && p.stack > 0 {
            return false;
        }
    }
    true
}
#[inline(always)]
fn round_open(s: &GameState) -> bool {
    for p in s.players.iter() {
        if p.status == STATUS_ACTIVE {
            let owe = (s.current_bet - p.bet).max(0);
            if p.rho < s.tau || owe > 0 {
                return true;
            }
        }
    }
    false
}
fn deal_next_street(s: &mut GameState) -> PyResult<()> {
    match s.round {
        ROUND_PREFLOP => {
            for _ in 0..3 {
                let c = s
                    .undealt
                    .pop()
                    .ok_or_else(|| PyValueError::new_err("deck underflow"))?;
                s.board.push(c);
            }
            s.round = ROUND_FLOP;
        }
        ROUND_FLOP => {
            let c = s
                .undealt
                .pop()
                .ok_or_else(|| PyValueError::new_err("deck underflow"))?;
            s.board.push(c);
            s.round = ROUND_TURN;
        }
        ROUND_TURN => {
            let c = s
                .undealt
                .pop()
                .ok_or_else(|| PyValueError::new_err("deck underflow"))?;
            s.board.push(c);
            s.round = ROUND_RIVER;
        }
        _ => return Err(PyValueError::new_err("No further streets to deal")),
    }
    Ok(())
}
fn reset_round(n: usize, s: &mut GameState) -> u64 {
    let mut changed: u64 = 0;
    for (idx, p) in s.players.iter_mut().enumerate() {
        if p.bet != 0 || (p.status == STATUS_ACTIVE && p.rho != -1_000_000_000) {
            changed |= 1 << idx;
        }
        p.bet = 0;
        if p.status == STATUS_ACTIVE {
            p.rho = -1_000_000_000;
        }
    }
    s.current_bet = 0;
    s.min_raise = s.bb;
    s.tau = 0;
    let mut x = (s.button + 1) % n;
    for _ in 0..n {
        if s.players[x].status == STATUS_ACTIVE {
            s.next_to_act = Some(x);
            return changed;
        }
        x = (x + 1) % n;
    }
    s.next_to_act = None;
    changed
}

#[inline(always)]
fn advance_next_player(n: usize, s: &mut GameState, i: usize) {
    let mut j = (i + 1) % n;
    for _ in 0..n {
        let pj = &s.players[j];
        if pj.status == STATUS_ACTIVE {
            let owej = (s.current_bet - pj.bet).max(0);
            if pj.rho < s.tau || owej > 0 {
                s.next_to_act = Some(j);
                return;
            }
        }
        j = (j + 1) % n;
    }
    s.next_to_act = None;
}

#[inline(always)]
fn add_chips(s: &mut GameState, idx: usize, amount: i32) -> PyResult<()> {
    if amount < 0 {
        return Err(PyValueError::new_err("negative amount"));
    }
    if s.players[idx].stack < amount {
        return Err(PyValueError::new_err("insufficient stack"));
    }
    s.players[idx].stack -= amount;
    s.players[idx].bet += amount;
    s.players[idx].cont += amount;
    s.pot += amount;
    Ok(())
}

fn settle_showdown(n: usize, s: &GameState) -> PyResult<Vec<i32>> {
    let mut active = [0usize; 6];
    let mut ac = 0;
    for (i, p) in s.players.iter().enumerate() {
        if p.status != STATUS_FOLDED {
            active[ac] = i;
            ac += 1;
        }
    }
    let mut levels = [0i32; 6];
    let mut lc = 0;
    for p in s.players.iter() {
        if p.cont > 0 {
            let mut found = false;
            for j in 0..lc {
                if levels[j] == p.cont {
                    found = true;
                    break;
                }
            }
            if !found {
                levels[lc] = p.cont;
                lc += 1;
            }
        }
    }
    if lc == 0 {
        return Ok(vec![0; n]);
    }
    levels[..lc].sort_unstable();

    let mut ranks = [HandRank::new(-1, [0; 5]); 6];
    for ai in 0..ac {
        let i = active[ai];
        let hole = s.players[i]
            .hole
            .ok_or_else(|| PyValueError::new_err("missing hole cards"))?;
        let mut seven = [0u8; 7];
        seven[0] = hole.0;
        seven[1] = hole.1;
        for (k, c) in s.board.iter().enumerate() {
            seven[2 + k] = *c;
        }
        ranks[i] = best5_rank_from_7(&seven);
    }

    let mut rewards = [0i32; 6];
    let mut y_prev = 0i32;
    let mut carry = 0i32;
    let mut last_winners = [0usize; 6];
    let mut last_wc = 0usize;

    for li in 0..lc {
        let y = levels[li];
        let mut contributors_count = 0i32;
        for p in s.players.iter() {
            if p.cont >= y {
                contributors_count += 1;
            }
        }
        let pk = contributors_count * (y - y_prev) + carry;
        let mut elig = [0usize; 6];
        let mut ec = 0usize;
        for ai in 0..ac {
            let i = active[ai];
            if s.players[i].cont >= y {
                elig[ec] = i;
                ec += 1;
            }
        }
        if ec > 0 {
            let mut best_rank = HandRank::new(-1, [0; 5]);
            for j in 0..ec {
                let idx = elig[j];
                if ranks[idx] > best_rank {
                    best_rank = ranks[idx];
                }
            }
            let mut winners = [0usize; 6];
            let mut wc = 0usize;
            for j in 0..ec {
                let idx = elig[j];
                if ranks[idx] == best_rank {
                    winners[wc] = idx;
                    wc += 1;
                }
            }
            last_wc = wc;
            last_winners[..wc].copy_from_slice(&winners[..wc]);
            let share = pk / wc as i32;
            let rem = pk % wc as i32;
            for j in 0..wc {
                rewards[winners[j]] += share;
            }
            if rem > 0 {
                let start = (s.button + 1) % n;
                let mut ordered = [0usize; 6];
                ordered[..wc].copy_from_slice(&winners[..wc]);
                ordered[..wc].sort_by_key(|j| ((*j + n) - start) % n);
                for k in 0..rem as usize {
                    rewards[ordered[k % wc]] += 1;
                }
            }
            carry = 0;
        } else {
            carry = pk;
        }
        y_prev = y;
    }

    if carry > 0 && last_wc > 0 {
        let share = carry / last_wc as i32;
        let rem = carry % last_wc as i32;
        for j in 0..last_wc {
            rewards[last_winners[j]] += share;
        }
        if rem > 0 {
            let start = (s.button + 1) % n;
            let mut ordered = [0usize; 6];
            ordered[..last_wc].copy_from_slice(&last_winners[..last_wc]);
            ordered[..last_wc].sort_by_key(|j| ((*j + n) - start) % n);
            for k in 0..rem as usize {
                rewards[ordered[k % last_wc]] += 1;
            }
        }
    }

    let mut rl = Vec::with_capacity(n);
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
    if p.status != STATUS_ACTIVE {
        return Ok(LegalActionInfo::new(vec![], None, None, None));
    }

    let owe = (s.current_bet - p.bet).max(0);
    let mut acts = Vec::<Action>::new();
    if owe > 0 {
        acts.push(Action {
            kind: 0,
            amount: None,
        }); // FOLD
    }
    if owe == 0 {
        acts.push(Action {
            kind: 1,
            amount: None,
        }); // CHECK
    }
    if owe > 0 {
        acts.push(Action {
            kind: 2,
            amount: None,
        }); // CALL
    }

    let can_raise = (p.status == STATUS_ACTIVE) && (p.stack > 0);
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
        acts.push(Action {
            kind: 3,
            amount: None,
        }); // RAISE_TO
        return Ok(LegalActionInfo::new(
            acts,
            Some(min_to),
            Some(max_to),
            Some(has_rr),
        ));
    }
    Ok(LegalActionInfo::new(acts, None, None, None))
}

#[inline(always)]
fn legal_actions_bits_from_state(s: &GameState) -> (u8, Option<i32>, Option<i32>, Option<bool>) {
    let i = match s.next_to_act {
        Some(x) => x,
        None => return (0, None, None, None),
    };
    let p = &s.players[i];
    if p.status != STATUS_ACTIVE {
        return (0, None, None, None);
    }

    let owe = (s.current_bet - p.bet).max(0);
    let mut mask: u8 = 0;
    if owe > 0 {
        mask |= 1;
    } // FOLD
    if owe == 0 {
        mask |= 2;
    } // CHECK
    if owe > 0 {
        mask |= 4;
    } // CALL

    let can_raise = (p.status == STATUS_ACTIVE) && (p.stack > 0);
    if !can_raise {
        return (mask, None, None, None);
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
        (mask, Some(min_to), Some(max_to), Some(has_rr))
    } else {
        (mask, None, None, None)
    }
}
fn advance_round_if_needed_internal(
    n: usize,
    s: &mut GameState,
) -> PyResult<(bool, Option<Vec<i32>>, u64)> {
    if let Some(lone) = one_survivor(s) {
        let mut rewards = vec![0i32; n];
        for i in 0..n {
            rewards[i] = if i == lone { s.pot } else { 0 } - s.players[i].cont;
        }
        return Ok((true, Some(rewards), 0));
    }

    if !round_open(s) {
        if (s.round == ROUND_PREFLOP || s.round == ROUND_FLOP || s.round == ROUND_TURN)
            && everyone_allin_or_folded(s)
        {
            while s.round != ROUND_RIVER {
                deal_next_street(s)?;
            }
            s.round = ROUND_SHOWDOWN;
            let rewards = settle_showdown(n, s)?;
            return Ok((true, Some(rewards), 0));
        }

        match s.round {
            ROUND_PREFLOP => {
                deal_next_street(s)?;
                let reset_mask = reset_round(n, s);
                return Ok((false, None, reset_mask));
            }
            ROUND_FLOP => {
                deal_next_street(s)?;
                let reset_mask = reset_round(n, s);
                return Ok((false, None, reset_mask));
            }
            ROUND_TURN => {
                deal_next_street(s)?;
                let reset_mask = reset_round(n, s);
                return Ok((false, None, reset_mask));
            }
            ROUND_RIVER => {
                s.round = ROUND_SHOWDOWN;
                let rewards = settle_showdown(n, s)?;
                return Ok((true, Some(rewards), 0));
            }
            _ => {}
        }
    }
    Ok((false, None, 0))
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
    deck: [u8; 52],
    cur: Option<Py<GameState>>,
    la_cache: (u8, Option<i32>, Option<i32>, Option<bool>),
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
            return Err(PyValueError::new_err("Engine fixed to 6 players per spec"));
        }
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let mut deck = [0u8; 52];
        let mut i = 0u8;
        while i < 52 {
            deck[i as usize] = i;
            i += 1;
        }
        Ok(Self {
            n: num_players,
            sb,
            bb,
            start_stack,
            rng,
            deck,
            cur: None,
            la_cache: (0, None, None, None),
        })
    }

    #[getter]
    #[inline(always)]
    fn N(&self) -> usize {
        self.n
    }

    /// Reset internal state and return a one-time full snapshot (Python builds its mirror from this)
    #[inline(always)]
    fn reset_hand(&mut self, py: Python<'_>, button: usize) -> PyResult<Py<GameState>> {
        let s = self.reset_hand_internal(button)?;
        let la = legal_actions_bits_from_state(&s);
        let py_s = Py::new(py, s)?;
        self.cur = Some(py_s.clone());
        self.la_cache = la;
        Ok(py_s)
    }

    /// Reset internal state and update the provided Python GameState mirror in-place.
    /// Returns None (mirror already updated).
    #[inline(always)]
    fn reset_hand_apply_py<'py>(
        &mut self,
        py: Python<'py>,
        py_state: &Bound<'py, pyo3::PyAny>,
        button: usize,
    ) -> PyResult<()> {
        let s_new = self.reset_hand_internal(button)?;
        let la = legal_actions_bits_from_state(&s_new);
        let py_state: Py<GameState> = py_state.extract()?;
        {
            let mut gs = py_state.borrow_mut(py);
            *gs = s_new;
        }
        self.cur = Some(py_state);
        self.la_cache = la;
        Ok(())
    }

    /// Optional: export full snapshot (debug)
    fn export_snapshot(&self, py: Python<'_>) -> PyResult<GameState> {
        match &self.cur {
            Some(s) => Ok(s.borrow(py).clone()),
            None => Err(PyValueError::new_err("no state")),
        }
    }

    /// Legal actions from current internal state
    fn legal_actions_now(&self, py: Python<'_>) -> PyResult<LegalActionInfo> {
        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let s = cell.borrow(py);
        legal_actions_from(&s)
    }

    /// Step using internal state; returns (done, rewards?, diff)
    fn step_diff(
        &mut self,
        py: Python<'_>,
        a: &Action,
    ) -> PyResult<(bool, Option<Vec<i32>>, StepDiff)> {
        // Snapshot BEFORE
        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let (board_len_before, round_before) = {
            let s = cell.borrow(py);
            (s.board.len(), s.round)
        };

        // Mutate
        let (done, rewards, last_log, changed_mask) = {
            let mut s_mut = cell.borrow_mut(py);
            let res = NLHEngine::step_on_internal(self.n, &mut s_mut, a)?;
            self.la_cache = legal_actions_bits_from_state(&s_mut);
            res
        };

        // AFTER
        let s2 = cell.borrow(py);
        let mut board_drawn: Vec<u8> = vec![];
        if s2.board.len() > board_len_before {
            board_drawn.extend_from_slice(&s2.board[board_len_before..]);
        }
        let round_label_change = if s2.round != round_before {
            Some(round_to_str(s2.round).to_string())
        } else {
            None
        };

        // Build player_updates using the changed mask
        let mut player_updates = Vec::<PlayerDiff>::new();
        let mut mask = changed_mask;
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            mask &= mask - 1; // clear lowest set bit
            let p_new = &s2.players[idx];
            player_updates.push(PlayerDiff {
                idx,
                stack: p_new.stack,
                bet: p_new.bet,
                cont: p_new.cont,
                rho: p_new.rho,
                status: Some(status_to_str(p_new.status).to_string()),
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

    /// Like step_diff but avoids constructing an Action on the Python side.
    /// kind: 0=FOLD,1=CHECK,2=CALL,3=RAISE_TO ; amount: None unless kind==3
    fn step_diff_raw(
        &mut self,
        py: Python<'_>,
        kind: u8,
        amount: Option<i32>,
    ) -> PyResult<(bool, Option<Vec<i32>>, StepDiff)> {
        let a = Action { kind, amount };
        self.step_diff(py, &a)
    }

    /// Advance round; returns (done, rewards?, diff)
    fn advance_round_if_needed_now(
        &mut self,
        py: Python<'_>,
    ) -> PyResult<(bool, Option<Vec<i32>>, StepDiff)> {
        // Snapshot BEFORE
        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let (board_len_before, round_before) = {
            let s = cell.borrow(py);
            (s.board.len(), s.round)
        };

        // Mutate
        let (done, rewards, round_reset_mask) = {
            let mut s_mut = cell.borrow_mut(py);
            let res = advance_round_if_needed_internal(self.n, &mut s_mut)?;
            self.la_cache = legal_actions_bits_from_state(&s_mut);
            res
        };

        // AFTER
        let s2 = cell.borrow(py);
        let mut board_drawn: Vec<u8> = vec![];
        if s2.board.len() > board_len_before {
            board_drawn.extend_from_slice(&s2.board[board_len_before..]);
        }
        let round_label_change = if s2.round != round_before {
            Some(round_to_str(s2.round).to_string())
        } else {
            None
        };

        // Build player_updates - include all players if round was reset (round_reset_mask != 0)
        let mut player_updates = Vec::<PlayerDiff>::new();
        if round_reset_mask != 0 {
            // Round was reset, include all players that were affected
            let mut mask = round_reset_mask;
            while mask != 0 {
                let idx = mask.trailing_zeros() as usize;
                mask &= mask - 1; // clear lowest set bit
                let p_new = &s2.players[idx];
                player_updates.push(PlayerDiff {
                    idx,
                    stack: p_new.stack,
                    bet: p_new.bet,
                    cont: p_new.cont,
                    rho: p_new.rho,
                    status: Some(status_to_str(p_new.status).to_string()),
                });
            }
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
    #[inline(always)]
    fn step_apply_py_raw<'py>(
        &mut self,
        py: Python<'py>,
        _py_state: &Bound<'py, pyo3::PyAny>,
        kind: u8,
        amount: Option<i32>,
    ) -> PyResult<(bool, Option<Vec<i32>>)> {
        let a = Action { kind, amount };
        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);
        let (done, rewards, _last_log, _changed_mask) =
            Self::step_on_internal(self.n, &mut s_mut, &a)?;
        self.la_cache = legal_actions_bits_from_state(&s_mut);
        Ok((done, rewards))
    }

    /// Fast path: advance round if needed, and update Python GameState mirror in-place.
    #[inline(always)]
    fn advance_round_if_needed_apply_py<'py>(
        &mut self,
        py: Python<'py>,
        _py_state: &Bound<'py, pyo3::PyAny>,
    ) -> PyResult<(bool, Option<Vec<i32>>)> {
        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);
        let (done, rewards, _mask) = advance_round_if_needed_internal(self.n, &mut s_mut)?;
        self.la_cache = legal_actions_bits_from_state(&s_mut);
        Ok((done, rewards))
    }

    /// Fast legal-actions: return (mask, min_to, max_to, has_rr)
    /// mask bits: 1=FOLD, 2=CHECK, 4=CALL, 8=RAISE_TO
    #[inline(always)]
    fn legal_actions_bits_now(&self) -> PyResult<(u8, Option<i32>, Option<i32>, Option<bool>)> {
        Ok(self.la_cache)
    }

    // ==============================
    // State setter functions for controlled variable modification
    // ==============================

    /// Set the pot value directly. Validates that the new pot value is non-negative.
    fn set_pot(&mut self, py: Python<'_>, new_pot: i32) -> PyResult<()> {
        if new_pot < 0 {
            return Err(PyValueError::new_err("pot cannot be negative"));
        }

        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);
        s_mut.pot = new_pot;
        Ok(())
    }

    /// Set the current bet value. Validates that it's non-negative.
    fn set_current_bet(&mut self, py: Python<'_>, new_current_bet: i32) -> PyResult<()> {
        if new_current_bet < 0 {
            return Err(PyValueError::new_err("current_bet cannot be negative"));
        }

        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);
        s_mut.current_bet = new_current_bet;
        // Update legal actions cache since current_bet affects available actions
        self.la_cache = legal_actions_bits_from_state(&s_mut);
        Ok(())
    }

    /// Set the minimum raise amount. Validates that it's positive.
    fn set_min_raise(&mut self, py: Python<'_>, new_min_raise: i32) -> PyResult<()> {
        if new_min_raise <= 0 {
            return Err(PyValueError::new_err("min_raise must be positive"));
        }

        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);
        s_mut.min_raise = new_min_raise;
        // Update legal actions cache since min_raise affects available actions
        self.la_cache = legal_actions_bits_from_state(&s_mut);
        Ok(())
    }

    /// Set the tau value (step index for raise rights).
    fn set_tau(&mut self, py: Python<'_>, new_tau: i64) -> PyResult<()> {
        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);
        s_mut.tau = new_tau;
        // Update legal actions cache since tau affects raise rights
        self.la_cache = legal_actions_bits_from_state(&s_mut);
        Ok(())
    }

    /// Set the step index.
    fn set_step_idx(&mut self, py: Python<'_>, new_step_idx: i64) -> PyResult<()> {
        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);
        s_mut.step_idx = new_step_idx;
        Ok(())
    }

    /// Set a player's stack. Validates that stack is non-negative and updates player status if needed.
    fn set_player_stack(
        &mut self,
        py: Python<'_>,
        player_idx: usize,
        new_stack: i32,
    ) -> PyResult<()> {
        if new_stack < 0 {
            return Err(PyValueError::new_err("stack cannot be negative"));
        }

        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);

        if player_idx >= s_mut.players.len() {
            return Err(PyValueError::new_err("player index out of range"));
        }

        s_mut.players[player_idx].stack = new_stack;

        // Update player status based on new stack
        if new_stack == 0
            && s_mut.players[player_idx].bet > 0
            && s_mut.players[player_idx].status != STATUS_FOLDED
        {
            s_mut.players[player_idx].status = STATUS_ALLIN;
        } else if new_stack > 0 && s_mut.players[player_idx].status == STATUS_ALLIN {
            s_mut.players[player_idx].status = STATUS_ACTIVE;
        }

        // Update legal actions cache since stack changes affect available actions
        self.la_cache = legal_actions_bits_from_state(&s_mut);
        Ok(())
    }

    /// Set a player's bet amount. Validates that bet is non-negative.
    fn set_player_bet(&mut self, py: Python<'_>, player_idx: usize, new_bet: i32) -> PyResult<()> {
        if new_bet < 0 {
            return Err(PyValueError::new_err("bet cannot be negative"));
        }

        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);

        if player_idx >= s_mut.players.len() {
            return Err(PyValueError::new_err("player index out of range"));
        }

        s_mut.players[player_idx].bet = new_bet;

        // Update legal actions cache since bet changes affect available actions
        self.la_cache = legal_actions_bits_from_state(&s_mut);
        Ok(())
    }

    /// Set a player's contribution (total amount put in pot this hand).
    fn set_player_cont(
        &mut self,
        py: Python<'_>,
        player_idx: usize,
        new_cont: i32,
    ) -> PyResult<()> {
        if new_cont < 0 {
            return Err(PyValueError::new_err("cont cannot be negative"));
        }

        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);

        if player_idx >= s_mut.players.len() {
            return Err(PyValueError::new_err("player index out of range"));
        }

        s_mut.players[player_idx].cont = new_cont;
        Ok(())
    }

    /// Set a player's rho value (step index when player last acted).
    fn set_player_rho(&mut self, py: Python<'_>, player_idx: usize, new_rho: i64) -> PyResult<()> {
        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);

        if player_idx >= s_mut.players.len() {
            return Err(PyValueError::new_err("player index out of range"));
        }

        s_mut.players[player_idx].rho = new_rho;

        // Update legal actions cache since rho affects raise rights and next_to_act
        self.la_cache = legal_actions_bits_from_state(&s_mut);
        Ok(())
    }

    /// Set a player's status. Validates status is one of: "active", "folded", "allin".
    fn set_player_status(
        &mut self,
        py: Python<'_>,
        player_idx: usize,
        new_status: String,
    ) -> PyResult<()> {
        if !["active", "folded", "allin"].contains(&new_status.as_str()) {
            return Err(PyValueError::new_err(
                "status must be 'active', 'folded', or 'allin'",
            ));
        }

        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);

        if player_idx >= s_mut.players.len() {
            return Err(PyValueError::new_err("player index out of range"));
        }

        s_mut.players[player_idx].status = str_to_status(&new_status);

        // Update legal actions cache since status changes affect available actions
        self.la_cache = legal_actions_bits_from_state(&s_mut);
        Ok(())
    }

    /// Convenience method to set multiple state variables at once with validation.
    /// This helps maintain state consistency when multiple related values need to change.
    fn set_state_batch(&mut self, py: Python<'_>, updates: HashMap<String, i64>) -> PyResult<()> {
        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);

        // Apply all updates to temporary values first, then validate
        let mut modified = false;

        for (key, value) in updates {
            match key.as_str() {
                "pot" => {
                    if value < 0 {
                        return Err(PyValueError::new_err("pot cannot be negative"));
                    }
                    s_mut.pot = value as i32;
                    modified = true;
                }
                "current_bet" => {
                    if value < 0 {
                        return Err(PyValueError::new_err("current_bet cannot be negative"));
                    }
                    s_mut.current_bet = value as i32;
                    modified = true;
                }
                "min_raise" => {
                    if value <= 0 {
                        return Err(PyValueError::new_err("min_raise must be positive"));
                    }
                    s_mut.min_raise = value as i32;
                    modified = true;
                }
                "tau" => {
                    s_mut.tau = value;
                    modified = true;
                }
                "step_idx" => {
                    s_mut.step_idx = value;
                    modified = true;
                }
                _ => {
                    let err_msg = format!("unknown state variable: {}", key);
                    return Err(PyValueError::new_err(err_msg));
                }
            }
        }

        if modified {
            // Update legal actions cache if any state that affects it was modified
            self.la_cache = legal_actions_bits_from_state(&s_mut);
        }

        Ok(())
    }

    /// Validate current state consistency and optionally fix issues.
    /// Returns warnings about inconsistencies found.
    fn validate_and_fix_state(
        &mut self,
        py: Python<'_>,
        fix_issues: bool,
    ) -> PyResult<Vec<String>> {
        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);
        let mut warnings = Vec::new();

        // Check pot consistency
        let calculated_pot: i32 = s_mut.players.iter().map(|p| p.cont).sum();
        if s_mut.pot != calculated_pot {
            warnings.push(format!(
                "Pot mismatch: pot={}, calculated from contributions={}",
                s_mut.pot, calculated_pot
            ));
            if fix_issues {
                s_mut.pot = calculated_pot;
            }
        }

        // Check current_bet consistency
        let max_bet = s_mut.players.iter().map(|p| p.bet).max().unwrap_or(0);
        if s_mut.current_bet != max_bet {
            warnings.push(format!(
                "Current bet mismatch: current_bet={}, max player bet={}",
                s_mut.current_bet, max_bet
            ));
            if fix_issues {
                s_mut.current_bet = max_bet;
            }
        }

        // Check player status consistency
        for (idx, player) in s_mut.players.iter_mut().enumerate() {
            // Player with zero stack and positive bet should be all-in
            if player.stack == 0 && player.bet > 0 && player.status == STATUS_ACTIVE {
                warnings.push(format!(
                    "Player {} has zero stack but is marked as active",
                    idx
                ));
                if fix_issues {
                    player.status = STATUS_ALLIN;
                }
            }

            // Player marked as all-in but has stack should be active
            if player.stack > 0 && player.status == STATUS_ALLIN {
                warnings.push(format!("Player {} has stack but is marked as all-in", idx));
                if fix_issues {
                    player.status = STATUS_ACTIVE;
                }
            }

            // Negative values checks
            if player.stack < 0 {
                warnings.push(format!(
                    "Player {} has negative stack: {}",
                    idx, player.stack
                ));
            }
            if player.bet < 0 {
                warnings.push(format!("Player {} has negative bet: {}", idx, player.bet));
            }
            if player.cont < 0 {
                warnings.push(format!(
                    "Player {} has negative contribution: {}",
                    idx, player.cont
                ));
            }
        }

        // Update legal actions cache after any fixes
        if fix_issues && !warnings.is_empty() {
            self.la_cache = legal_actions_bits_from_state(&s_mut);
        }

        Ok(warnings)
    }

    /// Set the board cards. Validates that all cards are in range 0-51 and no duplicates exist.
    /// Also updates the undealt deck to exclude board cards.
    fn set_board(&mut self, py: Python<'_>, new_board: Vec<u8>) -> PyResult<()> {
        // Validate card range
        for &card in &new_board {
            if card > 51 {
                return Err(PyValueError::new_err(format!(
                    "invalid card value: {}",
                    card
                )));
            }
        }

        // Check for duplicates in board
        let mut seen = HashSet::new();
        for &card in &new_board {
            if !seen.insert(card) {
                return Err(PyValueError::new_err(format!(
                    "duplicate card in board: {}",
                    card
                )));
            }
        }

        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);

        // Check for conflicts with existing hole cards
        for player in &s_mut.players {
            if let Some((c1, c2)) = player.hole {
                if new_board.contains(&c1) {
                    return Err(PyValueError::new_err(format!(
                        "board conflicts with player hole card: {}",
                        c1
                    )));
                }
                if new_board.contains(&c2) {
                    return Err(PyValueError::new_err(format!(
                        "board conflicts with player hole card: {}",
                        c2
                    )));
                }
            }
        }

        // Update board
        s_mut.board = new_board.clone();

        // Update undealt deck - remove board cards and all hole cards
        let mut used_cards = HashSet::new();
        used_cards.extend(&new_board);

        for player in &s_mut.players {
            if let Some((c1, c2)) = player.hole {
                used_cards.insert(c1);
                used_cards.insert(c2);
            }
        }

        s_mut.undealt = (0u8..52u8)
            .filter(|card| !used_cards.contains(card))
            .collect();

        Ok(())
    }

    /// Set a player's hole cards. Validates cards are in range, no duplicates, and no conflicts.
    fn set_player_hole(
        &mut self,
        py: Python<'_>,
        player_idx: usize,
        hole_cards: Option<(u8, u8)>,
    ) -> PyResult<()> {
        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);

        if player_idx >= s_mut.players.len() {
            return Err(PyValueError::new_err("player index out of range"));
        }

        if let Some((c1, c2)) = hole_cards {
            // Validate card range
            if c1 > 51 || c2 > 51 {
                return Err(PyValueError::new_err("invalid card value (must be 0-51)"));
            }

            // Check for duplicate cards in this player's hand
            if c1 == c2 {
                return Err(PyValueError::new_err(
                    "player cannot have duplicate hole cards",
                ));
            }

            // Check for conflicts with board
            if s_mut.board.contains(&c1) || s_mut.board.contains(&c2) {
                return Err(PyValueError::new_err("hole card conflicts with board"));
            }

            // Check for conflicts with other players' hole cards
            for (idx, player) in s_mut.players.iter().enumerate() {
                if idx != player_idx {
                    if let Some((other_c1, other_c2)) = player.hole {
                        if c1 == other_c1 || c1 == other_c2 || c2 == other_c1 || c2 == other_c2 {
                            return Err(PyValueError::new_err(format!(
                                "hole card conflicts with player {}",
                                idx
                            )));
                        }
                    }
                }
            }
        }

        // Update player's hole cards
        s_mut.players[player_idx].hole = hole_cards;

        // Update undealt deck
        let mut used_cards = HashSet::new();
        used_cards.extend(&s_mut.board);

        for player in &s_mut.players {
            if let Some((c1, c2)) = player.hole {
                used_cards.insert(c1);
                used_cards.insert(c2);
            }
        }

        s_mut.undealt = (0u8..52u8)
            .filter(|card| !used_cards.contains(card))
            .collect();

        Ok(())
    }

    /// Set the undealt deck directly. Validates that all cards are in range and no duplicates exist.
    /// This is useful for controlling the exact order of future cards.
    fn set_undealt(&mut self, py: Python<'_>, new_undealt: Vec<u8>) -> PyResult<()> {
        // Validate card range
        for &card in &new_undealt {
            if card > 51 {
                return Err(PyValueError::new_err(format!(
                    "invalid card value: {}",
                    card
                )));
            }
        }

        // Check for duplicates in undealt
        let mut seen = HashSet::new();
        for &card in &new_undealt {
            if !seen.insert(card) {
                return Err(PyValueError::new_err(format!(
                    "duplicate card in undealt: {}",
                    card
                )));
            }
        }

        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);

        // Check for conflicts with board and hole cards
        let mut used_cards = HashSet::new();
        used_cards.extend(&s_mut.board);

        for player in &s_mut.players {
            if let Some((c1, c2)) = player.hole {
                used_cards.insert(c1);
                used_cards.insert(c2);
            }
        }

        for &card in &new_undealt {
            if used_cards.contains(&card) {
                return Err(PyValueError::new_err(format!(
                    "undealt card {} conflicts with cards in play",
                    card
                )));
            }
        }

        s_mut.undealt = new_undealt;
        Ok(())
    }

    /// Convenience method to set up a complete card scenario with board, hole cards, and remaining deck.
    /// This ensures all cards are properly distributed with no conflicts.
    fn set_card_scenario(
        &mut self,
        py: Python<'_>,
        board: Vec<u8>,
        hole_cards: Vec<Option<(u8, u8)>>,
        undealt: Option<Vec<u8>>,
    ) -> PyResult<()> {
        let cell = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
        let mut s_mut = cell.borrow_mut(py);

        if hole_cards.len() != s_mut.players.len() {
            return Err(PyValueError::new_err(
                "hole_cards length must match number of players",
            ));
        }

        // Collect all cards that will be used
        let mut used_cards = HashSet::new();

        // Validate and collect board cards
        for &card in &board {
            if card > 51 {
                return Err(PyValueError::new_err(format!(
                    "invalid board card: {}",
                    card
                )));
            }
            if !used_cards.insert(card) {
                return Err(PyValueError::new_err(format!("duplicate card: {}", card)));
            }
        }

        // Validate and collect hole cards
        for (idx, hole) in hole_cards.iter().enumerate() {
            if let Some((c1, c2)) = hole {
                if *c1 > 51 || *c2 > 51 {
                    return Err(PyValueError::new_err(format!(
                        "invalid hole card for player {}",
                        idx
                    )));
                }
                if c1 == c2 {
                    return Err(PyValueError::new_err(format!(
                        "player {} cannot have duplicate hole cards",
                        idx
                    )));
                }
                if !used_cards.insert(*c1) {
                    return Err(PyValueError::new_err(format!("duplicate card: {}", c1)));
                }
                if !used_cards.insert(*c2) {
                    return Err(PyValueError::new_err(format!("duplicate card: {}", c2)));
                }
            }
        }

        // Handle undealt cards
        let final_undealt = if let Some(undealt_cards) = undealt {
            // Validate provided undealt cards
            for &card in &undealt_cards {
                if card > 51 {
                    return Err(PyValueError::new_err(format!(
                        "invalid undealt card: {}",
                        card
                    )));
                }
                if used_cards.contains(&card) {
                    return Err(PyValueError::new_err(format!(
                        "undealt card {} conflicts with cards in play",
                        card
                    )));
                }
            }

            // Check for duplicates in undealt
            let mut seen = HashSet::new();
            for &card in &undealt_cards {
                if !seen.insert(card) {
                    return Err(PyValueError::new_err(format!(
                        "duplicate card in undealt: {}",
                        card
                    )));
                }
            }

            undealt_cards
        } else {
            // Auto-generate remaining undealt cards
            (0u8..52u8)
                .filter(|card| !used_cards.contains(card))
                .collect()
        };

        // Apply all changes
        s_mut.board = board;
        for (idx, hole) in hole_cards.into_iter().enumerate() {
            s_mut.players[idx].hole = hole;
        }
        s_mut.undealt = final_undealt;

        Ok(())
    }
}

// ---- reset_hand for NLHEngine (mutates self.cur only through return) ----
impl NLHEngine {
    fn reset_hand_internal(&mut self, button: usize) -> PyResult<GameState> {
        let deck = &mut self.deck;
        for i in (1..52).rev() {
            let j = self.rng.gen_range(0..=i);
            deck.swap(i, j);
        }
        let template = PlayerState {
            hole: None,
            stack: self.start_stack,
            bet: 0,
            cont: 0,
            status: STATUS_ACTIVE,
            rho: -1_000_000_000,
        };
        let mut players: [PlayerState; 6] = core::array::from_fn(|_| template.clone());
        let mut undealt_idx = 52usize;
        for i in 0..self.n {
            undealt_idx -= 1;
            let c1 = deck[undealt_idx];
            undealt_idx -= 1;
            let c2 = deck[undealt_idx];
            players[i].hole = Some((c1, c2));
        }
        let board: Vec<u8> = Vec::with_capacity(5);
        let undealt = deck[..undealt_idx].to_vec();

        // blinds
        let sb_seat = (button + 1) % self.n;
        let bb_seat = (button + 2) % self.n;
        let sb_amt = self.sb.min(players[sb_seat].stack);
        players[sb_seat].stack -= sb_amt;
        players[sb_seat].bet = sb_amt;
        players[sb_seat].cont = sb_amt;
        if players[sb_seat].stack == 0 && sb_amt > 0 {
            players[sb_seat].status = STATUS_ALLIN;
        }
        let bb_amt = self.bb.min(players[bb_seat].stack);
        players[bb_seat].stack -= bb_amt;
        players[bb_seat].bet = bb_amt;
        players[bb_seat].cont = bb_amt;
        if players[bb_seat].stack == 0 && bb_amt > 0 {
            players[bb_seat].status = STATUS_ALLIN;
        }

        let current_bet = self.bb;
        let pot = sb_amt + bb_amt;

        Ok(GameState {
            button,
            round: ROUND_PREFLOP,
            board,
            undealt,
            players: Vec::from(players),
            current_bet,
            min_raise: self.bb,
            tau: 0,
            next_to_act: Some((button + 3) % self.n),
            step_idx: 0,
            pot,
            sb: self.sb,
            bb: self.bb,
            actions_log: Vec::with_capacity(64),
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
        if s.players[i].status != STATUS_ACTIVE {
            return Err(PyValueError::new_err("player not active"));
        }
        s.step_idx += 1;

        let mut changed: u64 = 0;

        let owe = (s.current_bet - s.players[i].bet).max(0);
        let b_old = s.current_bet;

        match a.kind {
            0 => {
                // FOLD
                changed |= 1 << i;
                s.players[i].status = STATUS_FOLDED;
                s.players[i].rho = s.step_idx;
                advance_next_player(n, s, i);
            }
            1 => {
                // CHECK
                if owe != 0 {
                    return Err(PyValueError::new_err("cannot CHECK when owing"));
                }
                changed |= 1 << i;
                s.players[i].rho = s.step_idx;
                advance_next_player(n, s, i);
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
                    s.players[i].status = STATUS_ALLIN;
                }
                s.players[i].rho = s.step_idx;
                advance_next_player(n, s, i);
            }
            3 => {
                // RAISE_TO
                changed |= 1 << i;
                let raise_to = a
                    .amount
                    .ok_or_else(|| PyValueError::new_err("RAISE_TO requires amount"))?;
                if raise_to <= s.current_bet {
                    return Err(PyValueError::new_err("raise_to must exceed current_bet"));
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
                    s.players[i].status = STATUS_ALLIN;
                }
                s.current_bet = raise_to;
                s.players[i].rho = s.step_idx;
                let full_inc = raise_to - b_old;
                if full_inc >= s.min_raise {
                    s.tau = s.step_idx;
                    s.min_raise = full_inc;
                    for j in 0..n {
                        if j != i && s.players[j].status == STATUS_ACTIVE {
                            changed |= 1 << j;
                            s.players[j].rho = -1_000_000_000;
                        }
                    }
                }
                advance_next_player(n, s, i);
            }
            _ => return Err(PyValueError::new_err("unknown action type")),
        }

        // log
        let aid = a.kind as i32;
        let rid = s.round as i32;
        let mut log_amt = 0i32;
        if a.kind == 3 {
            log_amt = s.current_bet;
        }
        s.actions_log.push((i, aid, log_amt, rid));

        let (done, rewards, round_reset_mask) = advance_round_if_needed_internal(n, s)?;
        // Merge the round reset changes with action-specific changes
        changed |= round_reset_mask;
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
    ];
    m.add("__all__", __all__)?;
    Ok(())
}
