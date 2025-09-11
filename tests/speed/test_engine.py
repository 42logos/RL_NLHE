import nlhe.core.engine as engine
import nlhe.core.rs_engine as rs_engine

from nlhe.core.types import Action, ActionType

import random
import pprint

class TestEngine:
    def __init__(self):
        self.engine = engine.NLHEngine(sb=1, bb=2, start_stack=100,rng=random.Random(1234))
        self.rs_engine = rs_engine.NLHEngine(sb=1, bb=2, start_stack=100,rng=random.Random(1234))
        
    def test_owned(self):
        initstate=self.engine.reset_hand()
        rs_initstate=self.rs_engine.reset_hand()
        
        
        owned=self.engine.owed(initstate,1)
        rs_owned=self.rs_engine.owed(initstate,1)
        
        print(initstate)
        
        s,_,_,_=self.engine.step(initstate,Action(ActionType.CALL))
        rss,_,_,_=self.rs_engine.step(initstate,Action(ActionType.CALL))
        assert s==rss
        
        lgi=self.engine.legal_actions(initstate)
        rslgi=self.rs_engine.legal_actions(initstate)
        print("legal actions",type(lgi),type(rslgi))
        assert lgi==rslgi
        
        lgi=self.engine.legal_actions(s)
        rslgi=self.rs_engine.legal_actions(rss)

        assert lgi==rslgi
        
        ac=random.choice(lgi.actions)
        print("chosen action",ac)
        
        if ac.kind==ActionType.RAISE_TO:
            ac=Action(ActionType.RAISE_TO,lgi.min_raise_to)
        
        s,_,_,_=self.engine.step(s,ac)
        rss,_,_,_=self.rs_engine.step(rss,ac)
        assert s==rss
        
        owned=self.engine.owed(s,3)
        rs_owned=self.rs_engine.owed(s,3)
        assert owned==rs_owned
        
        for i in range(60):
            
            
            if s.next_to_act is None:
                done, rw = self.engine.advance_round_if_needed(s)
                done_rs, rw_rs = self.rs_engine.advance_round_if_needed(rss)
                assert done==done_rs
                assert rw==rw_rs
                if done:
                    print("done",rw)
                    break
                
            lgi=self.engine.legal_actions(s)
            rslgi=self.rs_engine.legal_actions(s)
            assert lgi==rslgi
            
            if len(lgi.actions)==0:
                rw = self.engine.advance_round_if_needed(s)
                rrw = self.rs_engine.advance_round_if_needed(rss)
                
                assert rw==rrw
                print("end of round, advanced",rw)
                break

            ac=random.choice(lgi.actions)
            print("chosen action",ac)
            if ac.kind==ActionType.RAISE_TO:
                ac=Action(ActionType.RAISE_TO,lgi.min_raise_to)
            
            s,_,_,_=self.engine.step(s,ac)
            rss,_,_,_=self.rs_engine.step(rss,ac)
            assert s==rss
            
            owned=self.engine.owed(s,3)
            rs_owned=self.rs_engine.owed(s,3)
            assert owned==rs_owned
            

if __name__ == "__main__":
    tester = TestEngine()
    tester.test_owned()
    print("All tests passed.")