import unittest

from eval.t1.planners import (
    GemmaPlanner,
    NoRepeatPlanner,
    OraclePlanner,
    RandomPlanner,
    RuleBasedPlanner,
    instruction_to_task,
)


CLEAR = {
    "open_drawer": "pull the handle to open the drawer",
    "move_slider_left": "move the door to the left",
    "turn_on_led": "turn on the led",
}
CANDIDATES = list(CLEAR)


class FakeReasoningLLM:
    def __init__(self, instruction):
        self.instruction = instruction

    def get_instruction(self, messages):
        return self.instruction, {"role": "assistant", "content": self.instruction}


class PlannerTests(unittest.TestCase):
    def test_instruction_parser_requires_unique_feasible_task(self):
        self.assertEqual(
            instruction_to_task(
                "Instruction: pull the handle to open the drawer.", CANDIDATES, CLEAR
            ),
            "open_drawer",
        )
        self.assertIsNone(instruction_to_task("do something else", CANDIDATES, CLEAR))
        self.assertEqual(
            instruction_to_task("Please open the drawer.", CANDIDATES, CLEAR),
            "open_drawer",
        )

    def test_random_ignores_reward_history(self):
        planner = RandomPlanner(CLEAR, seed=4)
        planner.reset(CANDIDATES)
        choices = []
        for _ in range(8):
            decision = planner.begin_trial(None, [])
            choices.append(decision.selected_task)
            planner.observe_trial(
                {
                    "selected_task": decision.selected_task,
                    "selected_task_completed": True,
                    "reward": 1,
                }
            )
        self.assertGreater(len(set(choices)), 1)
        self.assertTrue(all(choice in CANDIDATES for choice in choices))

    def test_no_repeat_uses_manifest_order(self):
        planner = NoRepeatPlanner(CLEAR)
        planner.reset(CANDIDATES)
        choices = []
        for _ in CANDIDATES:
            decision = planner.begin_trial(None, [])
            choices.append(decision.selected_task)
            planner.observe_trial(
                {
                    "selected_task": decision.selected_task,
                    "selected_task_completed": True,
                    "reward": 0,
                }
            )
        self.assertEqual(choices, CANDIDATES)

    def test_rule_based_eliminates_completed_negative(self):
        planner = RuleBasedPlanner(CLEAR)
        planner.reset(CANDIDATES)
        first = planner.begin_trial(None, [])
        planner.observe_trial(
            {
                "selected_task": first.selected_task,
                "completed_tasks": [first.selected_task],
                "reward": 0,
            }
        )
        second = planner.begin_trial(None, [])
        self.assertNotEqual(second.selected_task, first.selected_task)
        self.assertIn(first.selected_task, second.metadata["negative_before"])

    def test_oracle_returns_latent_task(self):
        planner = OraclePlanner(CLEAR)
        planner.reset(CANDIDATES, meta_task="turn_on_led")
        self.assertEqual(planner.begin_trial(None, []).selected_task, "turn_on_led")

    def test_gemma_does_not_receive_meta_task_and_flags_invalid_output(self):
        planner = GemmaPlanner(CLEAR, FakeReasoningLLM("dance"))
        planner.reset(
            CANDIDATES,
            meta_task="turn_on_led",
            environment_description=[{"role": "user", "content": "scene"}],
        )
        decision = planner.begin_trial({"role": "user", "content": "trial 1"}, [])
        self.assertFalse(decision.valid)
        self.assertIsNone(decision.selected_task)
        self.assertNotIn("turn_on_led", str(planner.messages[0]))


if __name__ == "__main__":
    unittest.main()
