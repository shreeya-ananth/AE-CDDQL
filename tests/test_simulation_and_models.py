"""
Unit test suite verifying simulation environment, task generation, and all schedulers.
"""
import unittest
import numpy as np
import torch

from simulation.cloud_environment import CloudSimEnvironment
from simulation.workload_generator import generate_paper_tasks, generate_multi_batch_workload
from algorithms.classical_schedulers import RandomScheduler, RoundRobinScheduler, GreedyEarliestFinishScheduler
from algorithms.q_learning import QLearningScheduler
from algorithms.cddql_baseline import BaselineCDDQLScheduler
from algorithms.ae_cddql import AECDQLScheduler


class TestSimulationAndModels(unittest.TestCase):
    def test_environment_baseline_state(self):
        env = CloudSimEnvironment(num_vms=4, state_mode="baseline", reward_mode="baseline")
        state = env.reset()
        self.assertEqual(env.state_dim, 3)
        self.assertEqual(len(state), 3)
        self.assertEqual(len(env.tasks), 9)
        
        # Test step
        next_state, reward, done, info = env.step(action=0)
        self.assertEqual(len(next_state), 3)
        self.assertIn("exec_time", info)
        self.assertIn("wait_time", info)
        self.assertIn("energy", info)
        self.assertFalse(done)

    def test_environment_energy_aware_state(self):
        env = CloudSimEnvironment(num_vms=4, state_mode="energy_aware", reward_mode="multi_objective")
        state = env.reset()
        self.assertEqual(env.state_dim, 5)
        self.assertEqual(len(state), 5)
        
        # Execute until done
        done = False
        step_count = 0
        while not done:
            action = step_count % 4
            _, reward, done, info = env.step(action)
            step_count += 1
            
        self.assertEqual(step_count, 9)
        metrics = env.get_episode_metrics()
        self.assertGreater(metrics["makespan"], 0.0)
        self.assertGreater(metrics["total_energy_joules"], 0.0)
        self.assertGreater(metrics["total_cost"], 0.0)

    def test_classical_schedulers(self):
        env = CloudSimEnvironment()
        schedulers = [
            RandomScheduler(num_vms=4, seed=42),
            RoundRobinScheduler(num_vms=4),
            GreedyEarliestFinishScheduler(num_vms=4)
        ]
        
        for sched in schedulers:
            state = env.reset()
            done = False
            while not done:
                action = sched.select_action(state, env=env)
                self.assertTrue(0 <= action < 4)
                state, _, done, _ = env.step(action)
            m = env.get_episode_metrics()
            self.assertEqual(m["num_tasks"], 9)

    def test_cddql_baseline(self):
        env = CloudSimEnvironment(state_mode="baseline", reward_mode="baseline")
        agent = BaselineCDDQLScheduler(state_dim=3, num_vms=4, batch_size=4)
        
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            
        self.assertGreater(len(agent.memory), 0)

    def test_ae_cddql(self):
        env = CloudSimEnvironment(state_mode="energy_aware", reward_mode="multi_objective")
        agent = AECDQLScheduler(state_dim=5, num_vms=4, batch_size=4, use_adaptive_epsilon=True)
        
        initial_eps = agent.epsilon
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            
        self.assertLess(agent.epsilon, initial_eps)


if __name__ == "__main__":
    unittest.main()
