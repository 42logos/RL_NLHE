use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::{Ordering};
use std::collections::HashMap;

// -----------------------------
// Card utilities (0..=51)
// -----------------------------
#[pyfunction]
fn rank_of(c: u8) -> PyResult<i32> {
    if c > 51 { return Err(PyValueError::new_err("card out of range")); }
    Ok(2 + (c as i32 % 13))
}

#[pyfunction]
fn suit_of(c: u8) -> PyResult<i32> {
    if c > 51 { return Err(PyValueError::new_err("card out of range")); }
    Ok((c / 13) as i32)
}

#[pyfunction]
fn make_deck() -> Vec<u8> {
    (0u8..52u8).collect()
}

// -----------------------------
// ActionType mapping (keep IDs)
// 0: FOLD, 1: CHECK, 2: CALL, 3: RAISE_TO
// -----------------------------
#[pyfunction]
fn action_type_id(kind: u8) -> PyResult<i32> {
    // accept the same mapping used in Python
    if kind <= 3 { Ok(kind as i32) } else { Err(PyValueError::new_err("invalid action kind")) }
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

// -----------------------------
// Types (mirroring your Python dataclasses)
// -----------------------------
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
    fn new(kind: u8, amount: Option<i32>) -> PyResult<Self> {
        if kind > 3 { return Err(PyValueError::new_err("invalid action kind")); }
        Ok(Self { kind, amount })
    }
}

#[pyclass]
#[derive(Clone)]
struct PlayerState {
    /// Optional two-card hole
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
    /// rho: ordering timestamp as in Python engine
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
    fn new(
        actions: Vec<Action>,
        min_raise_to: Option<i32>,
        max_raise_to: Option<i32>,
        has_raise_right: Option<bool>,
    ) -> Self {
        Self { actions, min_raise_to, max_raise_to, has_raise_right }
    }
}

// -----------------------------
// Hand evaluation (Rust port of your Python logic)
// -----------------------------
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
        // same ordering semantics as Python tuple comparison
        match self.cat.cmp(&other.cat) {
            Ordering::Equal => self.tiebreak.cmp(&other.tiebreak),
            o => o,
        }
    }
}
impl PartialOrd for HandRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

fn ranks_of(cards5: &[u8]) -> Vec<i32> {
    let mut v: Vec<i32> = cards5.iter().map(|&c| 2 + (c as i32 % 13)).collect();
    v.sort_by(|a,b| b.cmp(a));
    v
}
fn suits_of(cards5: &[u8]) -> Vec<i32> {
    cards5.iter().map(|&c| (c / 13) as i32).collect()
}

fn straight_high(mut uniq: Vec<i32>) -> Option<i32> {
    uniq.sort_by(|a,b| b.cmp(a));
    // Wheel A-2-3-4-5
    let set: std::collections::HashSet<_> = uniq.iter().cloned().collect();
    if [14,5,4,3,2].iter().all(|x| set.contains(x)) {
        return Some(5);
    }
    if uniq.len() < 5 { return None; }
    for i in 0..=uniq.len()-5 {
        let w = &uniq[i..i+5];
        let diff = w[0] - w[4];
        let all_distinct = {
            let mut h = std::collections::HashSet::new();
            w.iter().all(|x| h.insert(*x))
        };
        if diff == 4 && all_distinct { return Some(w[0]); }
    }
    None
}

fn hand_rank_5(cards5: &[u8;5]) -> HandRank {
    let ranks = ranks_of(cards5);
    let suits = suits_of(cards5);

    let mut cnt: HashMap<i32, i32> = HashMap::new();
    for r in &ranks { *cnt.entry(*r).or_insert(0) += 1; }
    let mut bycnt: Vec<(i32,i32)> = cnt.into_iter().collect();
    bycnt.sort_by(|a,b| (b.1, b.0).cmp(&(a.1, a.0)));

    let is_flush = {
        let mut s = std::collections::HashSet::new();
        for v in suits { s.insert(v); }
        s.len() == 1
    };
    let mut uniq: Vec<i32> = ranks.clone();
    uniq.dedup(); // ranks are sorted desc, duplicates adjacent

    let s_high = straight_high(uniq.clone());

    if is_flush && s_high.is_some() {
        return HandRank { cat: HandCategory::StraightFlush as i32, tiebreak: vec![s_high.unwrap()] };
    }
    if bycnt[0].1 == 4 {
        let quad = bycnt[0].0;
        let kicker = ranks.iter().cloned().filter(|r| *r != quad).max().unwrap();
        return HandRank { cat: HandCategory::Four as i32, tiebreak: vec![quad, kicker] };
    }
    if bycnt[0].1 == 3 && bycnt[1].1 == 2 {
        let trips = bycnt[0].0; let pair = bycnt[1].0;
        return HandRank { cat: HandCategory::FullHouse as i32, tiebreak: vec![trips, pair] };
    }
    if is_flush {
        return HandRank { cat: HandCategory::Flush as i32, tiebreak: ranks.clone() };
    }
    if let Some(h) = s_high {
        return HandRank { cat: HandCategory::Straight as i32, tiebreak: vec![h] };
    }
    if bycnt[0].1 == 3 {
        let trips = bycnt[0].0;
        let kickers: Vec<i32> = ranks.iter().cloned().filter(|r| *r != trips).take(2).collect();
        return HandRank { cat: HandCategory::Trips as i32, tiebreak: [vec![trips], kickers].concat() };
    }
    if bycnt[0].1 == 2 && bycnt[1].1 == 2 {
        let hp = bycnt[0].0.max(bycnt[1].0);
        let lp = bycnt[0].0.min(bycnt[1].0);
        let kicker = ranks.iter().cloned().filter(|r| *r != hp && *r != lp).max().unwrap();
        return HandRank { cat: HandCategory::TwoPair as i32, tiebreak: vec![hp, lp, kicker] };
    }
    if bycnt[0].1 == 2 {
        let pair = bycnt[0].0;
        let kickers: Vec<i32> = ranks.iter().cloned().filter(|r| *r != pair).take(3).collect();
        return HandRank { cat: HandCategory::OnePair as i32, tiebreak: [vec![pair], kickers].concat() };
    }
    HandRank { cat: HandCategory::High as i32, tiebreak: ranks }
}

/// Python-friendly entry used by your eval.py
#[pyfunction]
fn best5_rank_from_7_py(cards7: Vec<u8>) -> PyResult<(i32, Vec<i32>)> {
    if cards7.len() != 7 { return Err(PyValueError::new_err("cards7 must have length 7")); }
    // choose all 21 5-card subsets
    let idxs = [
        [0,1,2,3,4],[0,1,2,3,5],[0,1,2,3,6],
        [0,1,2,4,5],[0,1,2,4,6],[0,1,2,5,6],
        [0,1,3,4,5],[0,1,3,4,6],[0,1,3,5,6],
        [0,1,4,5,6],[0,2,3,4,5],[0,2,3,4,6],
        [0,2,3,5,6],[0,2,4,5,6],[0,3,4,5,6],
        [1,2,3,4,5],[1,2,3,4,6],[1,2,3,5,6],
        [1,2,4,5,6],[1,3,4,5,6],[2,3,4,5,6],
    ];
    let mut best = HandRank { cat: -1, tiebreak: vec![] };
    for comb in idxs {
        let hand = [cards7[comb[0]], cards7[comb[1]], cards7[comb[2]], cards7[comb[3]], cards7[comb[4]]];
        let r = hand_rank_5(&hand);
        if r > best { best = r; }
    }
    Ok((best.cat, best.tiebreak))
}

// -----------------------------
// Engine
// -----------------------------
#[pyclass]
struct NLHEngine {
    n: usize,
    sb: i32,
    bb: i32,
    start_stack: i32,
    rng: StdRng,
}

#[pymethods]
impl NLHEngine {
    #[new]
    #[pyo3(signature = (sb=1, bb=2, start_stack=100, num_players=6, seed=None))]
    fn new(sb: i32, bb: i32, start_stack: i32, num_players: usize, seed: Option<u64>) -> PyResult<Self> {
        if num_players != 6 {
            return Err(PyValueError::new_err("Engine fixed to 6 players per spec"));
        }
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Ok(Self { n: num_players, sb, bb, start_stack, rng })
    }

    /// reset_hand(button=0) -> GameState
    fn reset_hand(&mut self, button: usize) -> PyResult<GameState> {
        let mut deck = make_deck();
        // shuffle (Fisher-Yates)
        for i in (1..deck.len()).rev() {
            let j = self.rng.gen_range(0..=i);
            deck.swap(i, j);
        }
        let mut players: Vec<PlayerState> = (0..self.n).map(|_| {
            PlayerState {
                hole: None,
                stack: self.start_stack,
                bet: 0, cont: 0,
                status: "active".into(),
                rho: -1_000_000_000
            }
        }).collect();

        for i in 0..self.n {
            let c1 = deck.pop().unwrap();
            let c2 = deck.pop().unwrap();
            players[i].hole = Some((c1, c2));
        }
        let board: Vec<u8> = Vec::new();
        let undealt = deck.clone();

        let sb_seat = (button + 1) % self.n;
        let bb_seat = (button + 2) % self.n;
        // SB
        let sb_amt = self.sb.min(players[sb_seat].stack);
        players[sb_seat].stack -= sb_amt; players[sb_seat].bet += sb_amt; players[sb_seat].cont += sb_amt;
        if players[sb_seat].stack == 0 && sb_amt > 0 { players[sb_seat].status = "allin".into(); }
        // BB
        let bb_amt = self.bb.min(players[bb_seat].stack);
        players[bb_seat].stack -= bb_amt; players[bb_seat].bet += bb_amt; players[bb_seat].cont += bb_amt;
        if players[bb_seat].stack == 0 && bb_amt > 0 { players[bb_seat].status = "allin".into(); }

        let current_bet = self.bb;
        let pot: i32 = players.iter().map(|p| p.cont).sum();

        let state = GameState {
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
        };
        Ok(state)
    }

    /// owed(s, i) -> int
    fn owed(&self, s: &GameState, i: usize) -> PyResult<i32> {
        Ok((s.current_bet - s.players[i].bet).max(0))
    }

    /// legal_actions(s) -> LegalActionInfo
    fn legal_actions(&self, s: &GameState) -> PyResult<LegalActionInfo> {
        let i = match s.next_to_act { Some(x) => x, None => return Ok(LegalActionInfo::new(vec![], None, None, None)) };
        let p = &s.players[i];
        if p.status != "active" {
            return Ok(LegalActionInfo::new(vec![], None, None, None));
        }
        let owe = (s.current_bet - p.bet).max(0);
        let mut acts = Vec::<Action>::new();
        if owe > 0 { acts.push(Action { kind: 0, amount: None }); } // FOLD
        if owe == 0 { acts.push(Action { kind: 1, amount: None }); } // CHECK
        if owe > 0 { acts.push(Action { kind: 2, amount: None }); } // CALL

        let can_raise = (p.status == "active") && (p.stack > 0);
        if !can_raise {
            return Ok(LegalActionInfo::new(acts, None, None, None));
        }

        let min_to = if s.current_bet == 0 { s.min_raise.max(1) } else { s.current_bet + s.min_raise };
        let max_to = p.bet + p.stack;
        let has_rr = (p.rho < s.tau) || (s.current_bet == 0);
        if max_to > s.current_bet {
            acts.push(Action { kind: 3, amount: None }); // RAISE_TO
            return Ok(LegalActionInfo::new(acts, Some(min_to), Some(max_to), Some(has_rr)));
        }
        Ok(LegalActionInfo::new(acts, None, None, None))
    }

    /// step(s, a) -> (s, done, rewards_or_none, info_dict)
    fn step(&self, py: Python<'_>, s: &mut GameState, a: &Action) -> PyResult<(GameState, bool, Option<Vec<i32>>, PyObject)> {
        let i = s.next_to_act.ok_or_else(|| PyValueError::new_err("no next_to_act"))?;
        if s.players[i].status != "active" {
            return Err(PyValueError::new_err("player not active"));
        }
        s.step_idx += 1;
        let mut advance_next = |s: &mut GameState, i: usize, n: usize| {
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
        let mut add_chips = |s: &mut GameState, idx: usize, amount: i32| -> PyResult<()> {
            if amount < 0 { return Err(PyValueError::new_err("negative amount")); }
            if s.players[idx].stack < amount { return Err(PyValueError::new_err("insufficient stack")); }
            s.players[idx].stack -= amount;
            s.players[idx].bet += amount;
            s.players[idx].cont += amount;
            Ok(())
        };

        let owe = (s.current_bet - s.players[i].bet).max(0);
        let b_old = s.current_bet;

        match a.kind {
            0 => { // FOLD
                s.players[i].status = "folded".into();
                s.players[i].rho = s.step_idx;
                advance_next(s, i, self.n);
            }
            1 => { // CHECK
                if owe != 0 { return Err(PyValueError::new_err("cannot CHECK when owing")); }
                s.players[i].rho = s.step_idx;
                advance_next(s, i, self.n);
            }
            2 => { // CALL
                if owe <= 0 { return Err(PyValueError::new_err("cannot CALL when owe==0")); }
                let call_amt = owe.min(s.players[i].stack);
                add_chips(s, i, call_amt)?;
                if s.players[i].stack == 0 { s.players[i].status = "allin".into(); }
                s.players[i].rho = s.step_idx;
                advance_next(s, i, self.n);
            }
            3 => { // RAISE_TO
                let raise_to = a.amount.ok_or_else(|| PyValueError::new_err("RAISE_TO requires amount"))?;
                if raise_to <= s.current_bet { return Err(PyValueError::new_err("raise_to must exceed current_bet")); }
                let max_to = s.players[i].bet + s.players[i].stack;
                if raise_to > max_to { return Err(PyValueError::new_err("raise exceeds max_to")); }
                let has_rr = (s.players[i].rho < s.tau) || (s.current_bet == 0);
                let required_min_to = if s.current_bet == 0 { s.min_raise.max(1) } else { s.current_bet + s.min_raise };
                if !has_rr && raise_to != max_to {
                    return Err(PyValueError::new_err("only all-in allowed; raise rights are closed"));
                } else if has_rr && !(raise_to >= required_min_to || raise_to == max_to) {
                    return Err(PyValueError::new_err("raise below required minimum"));
                }
                let delta = raise_to - s.players[i].bet;
                add_chips(s, i, delta)?;
                if s.players[i].stack == 0 { s.players[i].status = "allin".into(); }
                s.current_bet = raise_to;
                s.players[i].rho = s.step_idx;
                let full_inc = raise_to - b_old;
                if full_inc >= s.min_raise {
                    s.tau = s.step_idx;
                    s.min_raise = full_inc;
                    for j in 0..self.n {
                        if j != i && s.players[j].status == "active" {
                            s.players[j].rho = -1_000_000_000;
                        }
                    }
                }
                advance_next(s, i, self.n);
            }
            _ => return Err(PyValueError::new_err("unknown action type")),
        }

        // log (amount compact like your engine)
        let aid = a.kind as i32;
        let rid = round_label_id(&s.round_label);
        let mut log_amt = 0i32;
        if a.kind == 3 { log_amt = s.current_bet; } // RAISE_TO logs target
        s.actions_log.push((i, aid, log_amt, rid));

        // recompute pot/current_bet
        s.pot = s.players.iter().map(|pl| pl.cont).sum();
        s.current_bet = s.players.iter().map(|pl| pl.bet).max().unwrap_or(0);

        let (done, rewards) = self.advance_round_if_needed_internal(s)?;
        let info = PyDict::new(py).into_py(py);
        Ok((s.clone(), done, rewards, info))
    }

    /// advance_round_if_needed(s) -> (done, rewards_or_none)
    fn advance_round_if_needed(&self, s: &mut GameState) -> PyResult<(bool, Option<Vec<i32>>)> {
        self.advance_round_if_needed_internal(s)
    }
}

// ----- internal helpers for NLHEngine -----
impl NLHEngine {
    fn one_survivor(&self, s: &GameState) -> Option<usize> {
        let alive: Vec<usize> = s.players.iter().enumerate()
            .filter(|(_,p)| p.status != "folded")
            .map(|(i,_)| i).collect();
        if alive.len() == 1 { Some(alive[0]) } else { None }
    }

    fn everyone_allin_or_folded(&self, s: &GameState) -> bool {
        s.players.iter().all(|p| p.status != "active" || p.stack == 0)
    }

    fn round_open(&self, s: &GameState) -> bool {
        for (_i,p) in s.players.iter().enumerate() {
            if p.status == "active" {
                let owe = (s.current_bet - p.bet).max(0);
                if p.rho < s.tau || owe > 0 { return true; }
            }
        }
        false
    }

    fn deal_next_street(&self, s: &mut GameState) -> PyResult<()> {
        match s.round_label.as_str() {
            "Preflop" => {
                let draw: Vec<u8> = (0..3).map(|_| s.undealt.pop().ok_or_else(|| PyValueError::new_err("deck underflow"))).collect::<Result<_,_>>()?;
                s.board.extend(draw);
                s.round_label = "Flop".into();
            }
            "Flop" => {
                let c = s.undealt.pop().ok_or_else(|| PyValueError::new_err("deck underflow"))?;
                s.board.push(c);
                s.round_label = "Turn".into();
            }
            "Turn" => {
                let c = s.undealt.pop().ok_or_else(|| PyValueError::new_err("deck underflow"))?;
                s.board.push(c);
                s.round_label = "River".into();
            }
            _ => return Err(PyValueError::new_err("No further streets to deal")),
        }
        Ok(())
    }

    fn reset_round(&self, s: &mut GameState) {
        for p in &mut s.players {
            p.bet = 0;
            if p.status == "active" { p.rho = -1_000_000_000; }
        }
        s.current_bet = 0; s.min_raise = s.bb; s.tau = 0;
        let mut n = (s.button + 1) % self.n;
        for _ in 0..self.n {
            if s.players[n].status == "active" {
                s.next_to_act = Some(n); return;
            }
            n = (n + 1) % self.n;
        }
        s.next_to_act = None;
    }

    fn settle_showdown(&self, s: &GameState) -> PyResult<Vec<i32>> {
        let a: Vec<usize> = s.players.iter().enumerate().filter(|(_,p)| p.status != "folded").map(|(i,_)| i).collect();
        let mut levels: Vec<i32> = s.players.iter().map(|p| p.cont).filter(|v| *v > 0).collect();
        levels.sort(); levels.dedup();
        if levels.is_empty() { return Ok(vec![0; self.n]); }

        let mut ranks: HashMap<usize, (i32, Vec<i32>)> = HashMap::new();
        for i in &a {
            let hole = s.players[*i].hole.ok_or_else(|| PyValueError::new_err("missing hole cards"))?;
            let mut seven = vec![hole.0, hole.1];
            seven.extend_from_slice(&s.board);
            let (cat, tb) = best5_rank_from_7_py(seven)?;
            ranks.insert(*i, (cat, tb));
        }

        let mut rewards = vec![0i32; self.n];
        let mut y_prev = 0i32;
        let mut carry = 0i32;
        let mut last_nonempty_winners: Option<Vec<usize>> = None;

        for y in levels {
            let contributors_count = s.players.iter().filter(|p| p.cont >= y).count() as i32;
            let pk = contributors_count * (y - y_prev) + carry;
            let elig: Vec<usize> = a.iter().cloned().filter(|i| s.players[*i].cont >= y).collect();

            if !elig.is_empty() {
                let best_val = elig.iter().map(|i| ranks.get(i).unwrap()).max_by(|x,y| {
                    match x.0.cmp(&y.0) {
                        Ordering::Equal => x.1.cmp(&y.1),
                        o => o,
                    }
                }).unwrap().clone();

                let winners: Vec<usize> = elig.into_iter().filter(|i| ranks[i] == best_val).collect();
                last_nonempty_winners = Some(winners.clone());
                let share = pk / winners.len() as i32;
                let mut rem = pk % winners.len() as i32;
                for w in &winners { rewards[*w] += share; }
                if rem > 0 {
                    let start = (s.button + 1) % self.n;
                    let mut ordered = winners.clone();
                    ordered.sort_by_key(|j| ((*j + self.n) - start) % self.n);
                    let len = ordered.len();
                    for k in 0..rem as usize { rewards[ordered[k % len]] += 1; }
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
                let mut rem = carry % winners.len() as i32;
                for w in winners.iter() { rewards[*w] += share; }
                if rem > 0 {
                    let start = (s.button + 1) % self.n;
                    let mut ordered = winners.clone();
                    ordered.sort_by_key(|j| ((*j + self.n) - start) % self.n);
                    let len = ordered.len();
                    for k in 0..rem as usize { rewards[ordered[k % len]] += 1; }
                }
            }
        }
        // Convert to net winnings (zero-sum)
        let mut rl: Vec<i32> = Vec::with_capacity(self.n);
        for i in 0..self.n { rl.push(rewards[i] - s.players[i].cont); }
        Ok(rl)
    }

    fn advance_round_if_needed_internal(&self, s: &mut GameState) -> PyResult<(bool, Option<Vec<i32>>)> {
        if let Some(lone) = self.one_survivor(&s) {
            let mut rewards = vec![0i32; self.n];
            for i in 0..self.n {
                rewards[i] = if i == lone { s.pot } else { 0 } - s.players[i].cont;
            }
            return Ok((true, Some(rewards)));
        }

        if !self.round_open(&s) {
            if (s.round_label == "Preflop" || s.round_label == "Flop" || s.round_label == "Turn")
                && self.everyone_allin_or_folded(&s)
            {
                while s.round_label != "River" {
                    self.deal_next_street(s)?;
                }
                s.round_label = "Showdown".into();
                let rewards = self.settle_showdown(s)?;
                return Ok((true, Some(rewards)));
            }

            match s.round_label.as_str() {
                "Preflop" => { self.deal_next_street(s)?; self.reset_round(s); return Ok((false, None)); }
                "Flop"    => { self.deal_next_street(s)?; self.reset_round(s); return Ok((false, None)); }
                "Turn"    => { self.deal_next_street(s)?; self.reset_round(s); return Ok((false, None)); }
                "River"   => {
                    s.round_label = "Showdown".into();
                    let rewards = self.settle_showdown(s)?;
                    return Ok((true, Some(rewards)));
                }
                _ => {}
            }
        }
        Ok((false, None))
    }
}

// -----------------------------
// Python module
// -----------------------------
#[pymodule]
fn nlhe_engine(py: Python, m: &PyModule) -> PyResult<()> {
    // types
    m.add_class::<Action>()?;
    m.add_class::<PlayerState>()?;
    m.add_class::<GameState>()?;
    m.add_class::<LegalActionInfo>()?;
    m.add_class::<NLHEngine>()?;

    // fns
    m.add_function(wrap_pyfunction!(rank_of, m)?)?;
    m.add_function(wrap_pyfunction!(suit_of, m)?)?;
    m.add_function(wrap_pyfunction!(make_deck, m)?)?;
    m.add_function(wrap_pyfunction!(action_type_id, m)?)?;
    m.add_function(wrap_pyfunction!(round_label_id, m)?)?;
    m.add_function(wrap_pyfunction!(best5_rank_from_7_py, m)?)?;

    // __all__ convenience
    let __all__ = vec![
        "Action","PlayerState","GameState","LegalActionInfo","NLHEngine",
        "rank_of","suit_of","make_deck","action_type_id","round_label_id",
        "best5_rank_from_7_py"
    ];
    m.add("__all__", __all__)?;
    Ok(())
}
