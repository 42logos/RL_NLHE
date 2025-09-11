use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::Py;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

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

#[derive(Clone, Eq, PartialEq)]
struct HandRank {
    cat: i32,
    tiebreak: Vec<i32>,
}

impl Ord for HandRank {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.cat.cmp(&other.cat) {
            Ordering::Equal => self.tiebreak.cmp(&other.tiebreak),
            o => o,
        }
    }
}
impl PartialOrd for HandRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn ranks_of(cards5: &[u8]) -> Vec<i32> {
    let mut v: Vec<i32> = cards5.iter().map(|&c| 2 + (c as i32 % 13)).collect();
    v.sort_by(|a, b| b.cmp(a));
    v
}
fn suits_of(cards5: &[u8]) -> Vec<i32> {
    cards5.iter().map(|&c| (c / 13) as i32).collect()
}

fn straight_high(mut uniq: Vec<i32>) -> Option<i32> {
    uniq.sort_by(|a, b| b.cmp(a));
    let set: HashSet<i32> = uniq.iter().cloned().collect();
    if [14, 5, 4, 3, 2].iter().all(|x| set.contains(x)) {
        return Some(5);
    }
    if uniq.len() < 5 {
        return None;
    }
    for i in 0..=uniq.len() - 5 {
        let w = &uniq[i..i + 5];
        let diff = w[0] - w[4];
        let mut h = HashSet::new();
        let all_distinct = w.iter().all(|x| h.insert(*x));
        if diff == 4 && all_distinct {
            return Some(w[0]);
        }
    }
    None
}

fn hand_rank_5(cards5: &[u8; 5]) -> HandRank {
    let ranks = ranks_of(cards5);
    let suits = suits_of(cards5);

    let mut cnt: HashMap<i32, i32> = HashMap::new();
    for r in &ranks {
        *cnt.entry(*r).or_insert(0) += 1;
    }
    let mut bycnt: Vec<(i32, i32)> = cnt.into_iter().collect();
    bycnt.sort_by(|a, b| (b.1, b.0).cmp(&(a.1, a.0)));

    let is_flush = {
        let mut s = HashSet::new();
        for v in suits {
            s.insert(v);
        }
        s.len() == 1
    };
    let mut uniq: Vec<i32> = ranks.clone();
    uniq.dedup();

    let s_high = straight_high(uniq.clone());

    if is_flush && s_high.is_some() {
        return HandRank {
            cat: HandCategory::StraightFlush as i32,
            tiebreak: vec![s_high.unwrap()],
        };
    }
    if bycnt[0].1 == 4 {
        let quad = bycnt[0].0;
        let kicker = ranks.iter().cloned().filter(|r| *r != quad).max().unwrap();
        return HandRank {
            cat: HandCategory::Four as i32,
            tiebreak: vec![quad, kicker],
        };
    }
    if bycnt[0].1 == 3 && bycnt[1].1 == 2 {
        let trips = bycnt[0].0;
        let pair = bycnt[1].0;
        return HandRank {
            cat: HandCategory::FullHouse as i32,
            tiebreak: vec![trips, pair],
        };
    }
    if is_flush {
        return HandRank {
            cat: HandCategory::Flush as i32,
            tiebreak: ranks.clone(),
        };
    }
    if let Some(h) = s_high {
        return HandRank {
            cat: HandCategory::Straight as i32,
            tiebreak: vec![h],
        };
    }
    if bycnt[0].1 == 3 {
        let trips = bycnt[0].0;
        let kickers: Vec<i32> = ranks
            .iter()
            .cloned()
            .filter(|r| *r != trips)
            .take(2)
            .collect();
        return HandRank {
            cat: HandCategory::Trips as i32,
            tiebreak: [vec![trips], kickers].concat(),
        };
    }
    if bycnt[0].1 == 2 && bycnt[1].1 == 2 {
        let hp = bycnt[0].0.max(bycnt[1].0);
        let lp = bycnt[0].0.min(bycnt[1].0);
        let kicker = ranks
            .iter()
            .cloned()
            .filter(|r| *r != hp && *r != lp)
            .max()
            .unwrap();
        return HandRank {
            cat: HandCategory::TwoPair as i32,
            tiebreak: vec![hp, lp, kicker],
        };
    }
    if bycnt[0].1 == 2 {
        let pair = bycnt[0].0;
        let kickers: Vec<i32> = ranks
            .iter()
            .cloned()
            .filter(|r| *r != pair)
            .take(3)
            .collect();
        return HandRank {
            cat: HandCategory::OnePair as i32,
            tiebreak: [vec![pair], kickers].concat(),
        };
    }
    HandRank {
        cat: HandCategory::High as i32,
        tiebreak: ranks,
    }
}

fn best5_rank_from_7_rust(cards7: &[u8; 7]) -> (i32, Vec<i32>) {
    let idxs = [
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
    let mut best = HandRank {
        cat: -1,
        tiebreak: vec![],
    };
    for comb in idxs {
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
    (best.cat, best.tiebreak)
}

/// Python-friendly entry (parity with your eval.py)
#[pyfunction]
fn best5_rank_from_7_py(cards7: Vec<u8>) -> PyResult<(i32, Vec<i32>)> {
    if cards7.len() != 7 {
        return Err(PyValueError::new_err("cards7 must have length 7"));
    }
    let mut a = [0u8; 7];
    a.copy_from_slice(&cards7);
    Ok(best5_rank_from_7_rust(&a))
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
fn reset_round(n: usize, s: &mut GameState) -> u64 {
    let mut changed: u64 = 0;
    for (idx, p) in s.players.iter_mut().enumerate() {
        if p.bet != 0 || (p.status == "active" && p.rho != -1_000_000_000) {
            changed |= 1 << idx;
        }
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
            return changed;
        }
        x = (x + 1) % n;
    }
    s.next_to_act = None;
    changed
}
fn settle_showdown(n: usize, s: &GameState) -> PyResult<Vec<i32>> {
    let a: Vec<usize> = s
        .players
        .iter()
        .enumerate()
        .filter(|(_, p)| p.status != "folded")
        .map(|(i, _)| i)
        .collect();
    let mut levels: Vec<i32> = s
        .players
        .iter()
        .map(|p| p.cont)
        .filter(|v| *v > 0)
        .collect();
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
        let mut seven = [0u8; 7];
        seven[0] = hole.0;
        seven[1] = hole.1;
        for (k, c) in s.board.iter().enumerate() {
            seven[2 + k] = *c;
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
        let elig: Vec<usize> = a
            .iter()
            .cloned()
            .filter(|i| s.players[*i].cont >= y)
            .collect();

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
        if (s.round_label == "Preflop" || s.round_label == "Flop" || s.round_label == "Turn")
            && everyone_allin_or_folded(s)
        {
            while s.round_label != "River" {
                deal_next_street(s)?;
            }
            s.round_label = "Showdown".into();
            let rewards = settle_showdown(n, s)?;
            return Ok((true, Some(rewards), 0));
        }

        match s.round_label.as_str() {
            "Preflop" => {
                deal_next_street(s)?;
                let reset_mask = reset_round(n, s);
                return Ok((false, None, reset_mask));
            }
            "Flop" => {
                deal_next_street(s)?;
                let reset_mask = reset_round(n, s);
                return Ok((false, None, reset_mask));
            }
            "Turn" => {
                deal_next_street(s)?;
                let reset_mask = reset_round(n, s);
                return Ok((false, None, reset_mask));
            }
            "River" => {
                s.round_label = "Showdown".into();
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
            return Err(PyValueError::new_err("Engine fixed to 6 players per spec"));
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
        py_state.setattr("board", board_py)?;

        // actions_log: clear
        let al_obj = py_state.getattr("actions_log")?;
        let al_py = al_obj.downcast::<PyList>()?;
        let empty_list2 = pyo3::types::PyList::empty_bound(py);
        al_py.set_slice(0, al_py.len(), &empty_list2)?;
        py_state.setattr("actions_log", al_py)?;

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
        py_state.setattr("players", players_py)?;

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
        let (done, rewards, last_log, changed_mask) = {
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

    /// Like step_diff but avoids constructing an Action on the Python side.
    /// kind: 0=FOLD,1=CHECK,2=CALL,3=RAISE_TO ; amount: None unless kind==3
    fn step_diff_raw(
        &mut self,
        kind: u8,
        amount: Option<i32>,
    ) -> PyResult<(bool, Option<Vec<i32>>, StepDiff)> {
        let a = Action { kind, amount };
        self.step_diff(&a)
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
        let (done, rewards, round_reset_mask) = {
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
                    status: Some(p_new.status.clone()),
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
        // --- optional fast path ---
        let fast_cell = py_state.extract::<Py<GameState>>().ok();
        let (board_len_before, round_before) = if fast_cell.is_some() {
            (0, String::new())
        } else {
            let s = self
                .cur
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("no state"))?;
            (s.board.len(), s.round_label.clone())
        };

        // --- MUTATE RUST STATE ---
        let (done, rewards, _last_log, changed_mask) = {
            let s_mut = self
                .cur
                .as_mut()
                .ok_or_else(|| PyValueError::new_err("no state"))?;
            Self::step_on_internal(self.n, s_mut, a)?
        };

        if let Some(cell) = fast_cell {
            let mut cell_ref = cell.borrow_mut(py);
            *cell_ref = self.cur.as_ref().unwrap().clone();
            return Ok((done, rewards));
        }

        // --- APPLY CHANGES TO PYTHON MIRROR ---
        let s2 = self.cur.as_ref().unwrap();
        let round_changed = s2.round_label != round_before;

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
        if round_changed {
            py_state.setattr("round_label", s2.round_label.clone())?;
        }

        // board: append any newly dealt cards
        if s2.board.len() > board_len_before {
            let board_obj = py_state.getattr("board")?;
            let board_py = board_obj.downcast::<PyList>()?;
            for &c in &s2.board[board_len_before..] {
                board_py.append(c)?;
            }
            py_state.setattr("board", board_py)?;
        }

        // actions_log: push last entry (always one new)
        if let Some((i, aid, amt, rid)) = s2.actions_log.last().cloned() {
            let al_obj = py_state.getattr("actions_log")?;
            let al_py = al_obj.downcast::<PyList>()?;
            let tup = PyTuple::new_bound(
                py,
                &[i.into_py(py), aid.into_py(py), amt.into_py(py), rid.into_py(py)],
            );
            al_py.append(tup)?;
            py_state.setattr("actions_log", al_py)?;
        }

        // players: update only changed ones using the bitmask
        let players_obj = py_state.getattr("players")?;
        let players_py = players_obj.downcast::<PyList>()?;

        if round_changed {
            for (idx, p_new) in s2.players.iter().enumerate() {
                let p_obj = players_py.get_item(idx)?;
                p_obj.setattr("stack", p_new.stack)?;
                p_obj.setattr("bet", p_new.bet)?;
                p_obj.setattr("cont", p_new.cont)?;
                p_obj.setattr("rho", p_new.rho)?;
                p_obj.setattr("status", p_new.status.clone())?;
            }
        } else {
            let mut mask = changed_mask;
            while mask != 0 {
                let idx = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let p_new = &s2.players[idx];
                let p_obj = players_py.get_item(idx)?;
                p_obj.setattr("stack", p_new.stack)?;
                p_obj.setattr("bet", p_new.bet)?;
                p_obj.setattr("cont", p_new.cont)?;
                p_obj.setattr("rho", p_new.rho)?;
                p_obj.setattr("status", p_new.status.clone())?;
            }
        }
        py_state.setattr("players", players_py)?;

        Ok((done, rewards))
    }

    /// Fast path: advance round if needed, and update Python GameState mirror in-place.
    fn advance_round_if_needed_apply_py<'py>(
        &mut self,
        py: Python<'py>,
        py_state: &Bound<'py, pyo3::PyAny>,
    ) -> PyResult<(bool, Option<Vec<i32>>)> {
        // --- optional fast path ---
        let fast_cell = py_state.extract::<Py<GameState>>().ok();
        let (board_len_before, round_before) = if fast_cell.is_some() {
            (0, String::new())
        } else {
            let s = self
                .cur
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("no state"))?;
            (s.board.len(), s.round_label.clone())
        };

        // --- MUTATE RUST STATE ---
        let (done, rewards, _round_reset_mask) = {
            let s_mut = self
                .cur
                .as_mut()
                .ok_or_else(|| PyValueError::new_err("no state"))?;
            advance_round_if_needed_internal(self.n, s_mut)?
        };

        if let Some(cell) = fast_cell {
            let mut cell_ref = cell.borrow_mut(py);
            *cell_ref = self.cur.as_ref().unwrap().clone();
            return Ok((done, rewards));
        }

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
            py_state.setattr("board", board_py)?;
        }

        // push last log if any
        if let Some((i, aid, amt, rid)) = s2.actions_log.last().cloned() {
            let al_obj = py_state.getattr("actions_log")?;
            let al_py = al_obj.downcast::<PyList>()?;
            let tup = PyTuple::new_bound(
                py,
                &[i.into_py(py), aid.into_py(py), amt.into_py(py), rid.into_py(py)],
            );
            al_py.append(tup)?;
            py_state.setattr("actions_log", al_py)?;
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
        py_state.setattr("players", players_py)?;

        Ok((done, rewards))
    }

    /// Fast legal-actions: return (mask, min_to, max_to, has_rr)
    /// mask bits: 1=FOLD, 2=CHECK, 4=CALL, 8=RAISE_TO
    fn legal_actions_bits_now(&self) -> PyResult<(u8, Option<i32>, Option<i32>, Option<bool>)> {
        let s = self
            .cur
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no state"))?;
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
        if owe > 0 {
            mask |= 1;
        } // FOLD
        if owe == 0 {
            mask |= 2;
        } // CHECK
        if owe > 0 {
            mask |= 4;
        } // CALL

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
            let c1 = deck
                .pop()
                .ok_or_else(|| PyValueError::new_err("deck underflow"))?;
            let c2 = deck
                .pop()
                .ok_or_else(|| PyValueError::new_err("deck underflow"))?;
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
