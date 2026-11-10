import unittest
from collections import deque
from types import SimpleNamespace

from eval.t3.recurrent_state import prepare_trial_boundary


class FakePolicy:
    def __init__(self):
        self.img_queue = deque([1, 2], maxlen=12)
        self.gripper_queue = deque([1], maxlen=12)
        self.state_queue = deque([1], maxlen=12)
        self.mask_queue = deque([1], maxlen=12)
        self.text_queue = deque([1], maxlen=12)
        self.feature_cache = object()
        self.dt_feat_cache = [1]
        head = SimpleNamespace(
            hidden_state=(object(), object()), history_memory=[1, 2], rnn_out=object()
        )
        self.model = SimpleNamespace(
            lang_encoder=SimpleNamespace(lm_head=head), sep_lm_head=False
        )


class RecurrentStateTest(unittest.TestCase):
    def test_persistent_preserves_recurrence_but_clears_input_queues(self):
        policy = FakePolicy()
        event = prepare_trial_boundary(policy, "persistent", 2)
        after = event["after"]
        self.assertTrue(after["heads"]["lang_encoder.lm_head"]["hidden_present"])
        self.assertEqual(
            after["heads"]["lang_encoder.lm_head"]["history_length"], 2
        )
        self.assertTrue(all(length == 0 for length in after["queue_lengths"].values()))
        self.assertFalse(event["recurrent_reset_applied"])

    def test_reset_clears_recurrence_and_input_queues(self):
        policy = FakePolicy()
        event = prepare_trial_boundary(policy, "reset", 2)
        after = event["after"]
        self.assertFalse(after["heads"]["lang_encoder.lm_head"]["hidden_present"])
        self.assertEqual(
            after["heads"]["lang_encoder.lm_head"]["history_length"], 0
        )
        self.assertFalse(after["heads"]["lang_encoder.lm_head"]["rnn_out_present"])
        self.assertTrue(event["recurrent_reset_applied"])


if __name__ == "__main__":
    unittest.main()
