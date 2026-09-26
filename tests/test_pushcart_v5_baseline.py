from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"HomieRL"))
from homie_pushcart_v5.baseline_acceptance import summarize_trial, task_command, rotate_xy


class BaselineAcceptanceTests(unittest.TestCase):
    def trace(self, task="forward"):
        t = np.arange(1,601)*.02
        ref = np.array([task_command(task, s) for s in t])
        return dict(time_s=t, actual=ref.copy(), reference=ref, cart_yaw=np.zeros(len(t)),
                    cart_com=np.cumsum(ref, axis=0)*.02)

    def test_perfect_following_passes_but_frozen_cart_cannot(self):
        trace = self.trace()
        self.assertTrue(summarize_trial(trace,"forward")["pass"])
        trace["actual"][:]=0
        result=summarize_trial(trace,"forward")
        self.assertFalse(result["pass"])
        self.assertFalse(result["progress_pass"])

    def test_filtered_success_never_replaces_raw_gate(self):
        trace=self.trace()
        trace["actual"][:,2]=.30*np.sin(2*np.pi*4*trace["time_s"])
        result=summarize_trial(trace,"forward")
        self.assertTrue(result["filtered_tracking_diagnostic_pass"])
        self.assertFalse(result["raw_tracking_pass"])
        self.assertFalse(result["pass"])

    def test_failure_or_truncated_run_cannot_pass(self):
        trace=self.trace()
        self.assertFalse(summarize_trial(trace,"forward",failed=True)["pass"])
        trace={k:v[:300] for k,v in trace.items()}
        self.assertFalse(summarize_trial(trace,"forward")["pass"])

    def test_stop_requires_actual_stopping_even_with_small_average_error(self):
        trace=self.trace("stop")
        trace["actual"][trace["time_s"]>=6,0]=.09
        result=summarize_trial(trace,"stop")
        self.assertTrue(result["raw_tracking_pass"])
        self.assertIsNone(result["stop_confirmed_after_switch_s"])
        self.assertFalse(result["pass"])

    def test_hold_drift_cannot_hide_behind_low_velocity(self):
        trace=self.trace("hold")
        trace["actual"][:,0]=.02
        trace["cart_com"][:,0]=trace["time_s"]*.02
        result=summarize_trial(trace,"hold")
        self.assertTrue(result["raw_tracking_pass"])
        self.assertFalse(result["pass"])

    def test_world_fixed_error_filter_does_not_treat_turn_as_vector_change(self):
        trace=self.trace()
        trace["cart_yaw"] = trace["time_s"]*.8
        world_error=np.tile([.02,0,0],(600,1))
        trace["actual"]+=rotate_xy(world_error,-trace["cart_yaw"])
        result=summarize_trial(trace,"forward")
        np.testing.assert_allclose(np.linalg.norm(result["filtered_twist_rmse"]),.02,atol=1e-10)

    def test_time_gaps_cannot_supply_a_fake_dwell(self):
        trace=self.trace("stop")
        trace={k:np.concatenate((v[:300],v[590:])) for k,v in trace.items()}
        self.assertEqual(summarize_trial(trace,"stop")["status"],"gapped_trace")

    def test_good_first_turn_cannot_hide_bad_reversal(self):
        trace=self.trace("turn_reversal")
        trace["actual"][trace["time_s"]>=6,2] *= .5
        result=summarize_trial(trace,"turn_reversal")
        self.assertGreater(result["gains"][2], .7)
        self.assertFalse(result["segments"]["after_switch"]["pass"])
        self.assertFalse(result["pass"])

    def test_brief_stop_then_rebound_cannot_pass(self):
        trace=self.trace("stop")
        trace["actual"][trace["time_s"]>=6.6,0]=.09
        result=summarize_trial(trace,"stop")
        self.assertTrue(result["raw_tracking_pass"])
        self.assertTrue(result["stop_rebounded_after_first_dwell"])
        self.assertIsNone(result["stop_confirmed_after_switch_s"])
        self.assertFalse(result["pass"])


if __name__=="__main__": unittest.main()
