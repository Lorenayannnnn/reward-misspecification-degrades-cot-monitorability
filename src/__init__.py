"""
Code for "Reward Misspecification Degrades Chain-of-Thought Monitorability". Per task a control
model optimizes the true reward and a misaligned model the misspecified reward. The hack label Y
and the monitor scores are computed offline by compute_monitorability.py.
"""
