#![allow(clippy::needless_return, clippy::many_single_char_names)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;

#[inline]
fn rank_of(c: u8) -> u8 {
    2 + (c % 13)
}
#[inline]
fn suit_of(c: u8) -> u8 {
    c / 13
}

#[derive(Clone, Debug)]
struct RankVal {
    cat: u8,
    tiebreak: Vec<u8>,
}

impl PartialEq for RankVal {
    fn eq(&self, other: &Self) -> bool {
        self.cat == other.cat && self.tiebreak == other.tiebreak
    }
}
impl Eq for RankVal {}
impl PartialOrd for RankVal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RankVal {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.cat.cmp(&other.cat) {
            Ordering::Equal => self.tiebreak.cmp(&other.tiebreak),
            o => o,
        }
    }
}

// Categories (must match Python)
const STRAIGHT_FLUSH: u8 = 8;
const FOUR: u8 = 7;
const FULL_HOUSE: u8 = 6;
const FLUSH: u8 = 5;
const STRAIGHT: u8 = 4;
const TRIPS: u8 = 3;
const TWO_PAIR: u8 = 2;
const ONE_PAIR: u8 = 1;
const HIGH: u8 = 0;

fn hand_rank_5(cards5: &[u8; 5]) -> RankVal {
    // ranks sorted desc
    let mut ranks: [u8; 5] = [0; 5];
    let mut suits: [u8; 5] = [0; 5];
    for i in 0..5 {
        let c = cards5[i];
        ranks[i] = rank_of(c);
        suits[i] = suit_of(c);
    }
    ranks.sort_unstable_by(|a, b| b.cmp(a));

    // flush?
    let is_flush = suits.iter().all(|&s| s == suits[0]);

    // frequency by rank (2..14)
    let mut freq = [0u8; 15];
    for &r in &ranks {
        freq[r as usize] += 1;
    }
    // collect (rank,count) sorted by count desc then rank desc
    let mut cnt: Vec<(u8, u8)> = Vec::with_capacity(5);
    let mut rr = 14u8;
    while rr >= 2 {
        let f = freq[rr as usize];
        if f > 0 {
            cnt.push((rr, f));
        }
        if rr == 2 {
            break;
        }
        rr -= 1;
    }
    cnt.sort_unstable_by(|a, b| match b.1.cmp(&a.1) {
        Ordering::Equal => b.0.cmp(&a.0),
        o => o,
    });

    // straight high (A can be low 5-high)
    let mut uniq: Vec<u8> = Vec::with_capacity(5);
    for &r in &ranks {
        if uniq.last().copied() != Some(r) {
            uniq.push(r);
        }
    }
    let mut s_high: Option<u8> = None;
    if uniq.len() >= 5 {
        for i in 0..=uniq.len() - 5 {
            if uniq[i] - uniq[i + 4] == 4 {
                s_high = Some(uniq[i]);
                break;
            }
        }
    }
    if s_high.is_none() {
        let has = |x: u8| uniq.iter().any(|&y| y == x);
        if has(14) && has(5) && has(4) && has(3) && has(2) {
            s_high = Some(5);
        }
    }

    // categories in order
    if is_flush {
        if let Some(h) = s_high {
            return RankVal {
                cat: STRAIGHT_FLUSH,
                tiebreak: vec![h],
            };
        }
    }

    if cnt[0].1 == 4 {
        let quad = cnt[0].0;
        let mut kicker = 0u8;
        for &r in &ranks {
            if r != quad {
                kicker = r;
                break;
            }
        }
        return RankVal {
            cat: FOUR,
            tiebreak: vec![quad, kicker],
        };
    }

    if cnt.len() >= 2 && cnt[0].1 == 3 && cnt[1].1 == 2 {
        return RankVal {
            cat: FULL_HOUSE,
            tiebreak: vec![cnt[0].0, cnt[1].0],
        };
    }

    if is_flush {
        return RankVal {
            cat: FLUSH,
            tiebreak: ranks.to_vec(),
        };
    }

    if let Some(h) = s_high {
        return RankVal {
            cat: STRAIGHT,
            tiebreak: vec![h],
        };
    }

    if cnt[0].1 == 3 {
        let trips = cnt[0].0;
        let mut kickers: Vec<u8> = Vec::with_capacity(2);
        for &r in &ranks {
            if r != trips {
                kickers.push(r);
                if kickers.len() == 2 {
                    break;
                }
            }
        }
        return RankVal {
            cat: TRIPS,
            tiebreak: vec![trips, kickers[0], kickers[1]],
        };
    }

    if cnt.len() >= 2 && cnt[0].1 == 2 && cnt[1].1 == 2 {
        let highp = cnt[0].0.max(cnt[1].0);
        let lowp = cnt[0].0.min(cnt[1].0);
        let mut kicker = 0u8;
        for &r in &ranks {
            if r != highp && r != lowp {
                kicker = r;
                break;
            }
        }
        return RankVal {
            cat: TWO_PAIR,
            tiebreak: vec![highp, lowp, kicker],
        };
    }

    if cnt[0].1 == 2 {
        let pair = cnt[0].0;
        let mut kickers: Vec<u8> = Vec::with_capacity(3);
        for &r in &ranks {
            if r != pair {
                kickers.push(r);
                if kickers.len() == 3 {
                    break;
                }
            }
        }
        return RankVal {
            cat: ONE_PAIR,
            tiebreak: vec![pair, kickers[0], kickers[1], kickers[2]],
        };
    }

    return RankVal {
        cat: HIGH,
        tiebreak: ranks.to_vec(),
    };
}


fn best5_from_7(cards7: &[u8]) -> RankVal {
    debug_assert_eq!(cards7.len(), 7);
    let mut best: Option<RankVal> = None;
    for a in 0..3 {
        for b in (a + 1)..4 {
            for c in (b + 1)..5 {
                for d in (c + 1)..6 {
                    for e in (d + 1)..7 {
                        let comb = [
                            cards7[a],
                            cards7[b],
                            cards7[c],
                            cards7[d],
                            cards7[e],
                        ];
                        let r = hand_rank_5(&comb);
                        if best.as_ref().map_or(true, |x| r > *x) {
                            best = Some(r);
                        }
                    }
                }
            }
        }
    }
    return best.unwrap();
}

#[pyfunction]
fn best5_rank_from_7_py(cards7: Vec<u8>) -> PyResult<(u8, Vec<u8>)> {
    if cards7.len() != 7 {
        return Err(PyValueError::new_err("cards7 must have length 7"));
    }
    let rv = best5_from_7(&cards7);
    Ok((rv.cat, rv.tiebreak))
}

#[pyfunction]
fn best5_rank_from_7_batch_py(
    py: Python<'_>,
    holes_flat: Vec<u8>,
    board: Vec<u8>,
) -> PyResult<Vec<(u8, Vec<u8>)>> {
    if board.len() != 5 {
        return Err(PyValueError::new_err("board must have length 5"));
    }
    if holes_flat.len() % 2 != 0 {
        return Err(PyValueError::new_err("holes_flat must have even length"));
    }
    let res = py.allow_threads(|| {
        let mut out: Vec<(u8, Vec<u8>)> = Vec::with_capacity(holes_flat.len() / 2);
        let mut i = 0usize;
        while i < holes_flat.len() {
            let h1 = holes_flat[i];
            let h2 = holes_flat[i + 1];
            let mut cards7 = Vec::with_capacity(7);
            cards7.push(h1);
            cards7.push(h2);
            cards7.extend_from_slice(&board);
            let rv = best5_from_7(&cards7);
            out.push((rv.cat, rv.tiebreak));
            i += 2;
        }
        out
    });
    Ok(res)
}

#[pymodule]
fn nlhe_eval(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(best5_rank_from_7_py, m)?)?;
    m.add_function(wrap_pyfunction!(best5_rank_from_7_batch_py, m)?)?;
    Ok(())
}
