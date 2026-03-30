from env.environment import ClinicalEnv
from env.grader import compute_score
from env.logger import EpisodeLogger
import copy

env = ClinicalEnv(difficulty="hard")
logger = EpisodeLogger()

state = env.reset()
done = False
step_count = 0

while not done:
    if state["complication"]:
        action = "handle_complication"
    else:
        action = "perform_step"

    next_state, reward, done, _ = env.step(action)

    logger.log(step_count, copy.deepcopy(state), action, reward)

    state = next_state
    step_count += 1

logger.save()

print("Final Score:", compute_score(state))
print("Episode saved to episode.json")