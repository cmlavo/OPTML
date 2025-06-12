"""
Here we define some adversarial examples schedulers.

These schedulers are classes with a k_min and k_max (the min and max k they can return).
As functions, they take as input the current epoch and total number of epochs,
and return the probability distribution of each k value for the current epoch as a dictionnary. 

k is the number of PGD attack iterations, so together with epsilon (perturbation size)
they define the difficulty of the adversarial example.

-k: the number of PGD iterations (attack strength via iterations).
- epsilon: the maximum perturbation magnitude (attack strength via amplitude).

Schedulers provide a unified interface:
    epsilon, k_distribution = scheduler(epoch, max_epochs)

where:
- epsilon is a float for this epochs perturbation size.
- k_distribution is a dict mapping iteratioon counts k to sampling probabilities.


All schedulers subclass BaseScheduler to share parameters and signature.
Its only for the CompositeScheduler that we need to define the epsilon strategy.
"""
import math
import random

# TODO: Implement the schedulers in the main

class BaseScheduler:
    """
    Abstract base class for K schedulers.

    Init args:
      - k_min, k_max: int bounds for PGD iterations.
      - epsilon_max: float constant epsilon returned by all K schedulers.

    Subclasses implement `_get_k_distribution`.
    """
    def __init__(self, k_min: int, k_max: int, epsilon_max: float = 0.3):
        assert k_min <= k_max, "k_min must be <= k_max"
        self.k_min = k_min
        self.k_max = k_max
        self.epsilon_max = epsilon_max

    def __call__(self, epoch: int, max_epochs: int):
        eps = self.epsilon_max
        k_dist = self._get_k_distribution(epoch, max_epochs)
        return eps, k_dist

    def _get_k_distribution(self, epoch: int, max_epochs: int) -> dict:
        raise NotImplementedError("Subclasses must implement k distribution")


class ConstantScheduler(BaseScheduler):
    """Always returns constant k = k_max."""
    def __init__(self, k_min: int, k_max: int, epsilon_max: float = 0.3):
        super().__init__(k_min, k_max, epsilon_max)
        
    def _get_k_distribution(self, epoch, max_epochs):
        return {self.k_max: 1.0}


class LinearScheduler(BaseScheduler):
    """k linearly increases k_min→k_max over epochs."""
    def _get_k_distribution(self, epoch, max_epochs):
        frac = epoch / max_epochs
        k = self.k_min + round(frac * (self.k_max - self.k_min))
        return {k: 1.0}


class LinearUniformMixScheduler(BaseScheduler):
    """
    The scheduler advocated by the CAT paper.

    k linearly increases from k_min (0 in the CAT paper) to k_max during the epochs.
    However, instead of only returning a proportion 1 of the chosen k (current_k), we return a uniform
    proportion of k_min, k_min + 1, ... ,current_k.
    Mix uniform over all k values from k_min to current linear k_max.
    """
    def _get_k_distribution(self, epoch, max_epochs):
        frac = epoch / max_epochs
        max_k = self.k_min + round(frac * (self.k_max - self.k_min))
        ks = list(range(self.k_min, max_k + 1))
        prob = 1.0/len(ks)
        return {k: prob for k in ks}


class ExponentialScheduler(BaseScheduler):
    """k grows exponentially k_min→k_max.  the expo is used to 
    create a non-linear ramp from k_min to k_max. """
    def _get_k_distribution(self, epoch, max_epochs):
        if self.k_min == 0:
            base = self.k_max + 1
            raw = 1 * (base ** (epoch / max_epochs))
            k = math.floor(raw) - 1
        else:
            ratio = self.k_max / self.k_min
            raw = self.k_min * (ratio ** (epoch / max_epochs))
            k = round(raw)
        k = max(self.k_min, min(self.k_max, k))
        return {k: 1.0}


"""
class CyclicScheduler(BaseScheduler):
    '''k oscillates cosinusoidally k_min to k_max to k_min.
      The cosine function is used to create a smooth transition between k_min and k_max.
      '''
    def _get_k_distribution(self, epoch, max_epochs):
        cosv = math.pi * epoch / max_epochs
        frac = (1 + math.cos(cosv)) / 2  # 1→0
        k = self.k_min + frac * (self.k_max - self.k_min)
        return {round(k): 1.0}
"""


# Fix issues of old one
class CyclicScheduler(BaseScheduler):
    """
    k oscillates cosinusoidally between k_min and k_max.
    Starts at k_min. Completes `cycles` full oscillations over max_epochs.
    """
    def __init__(self, k_min, k_max, cycles=2, epsilon_max=0.3):
        super().__init__(k_min, k_max, epsilon_max)
        self.cycles = cycles

    def _get_k_distribution(self, epoch, max_epochs):
        # Phase shift by pi to start at k_min
        cosv = 2 * math.pi * self.cycles * epoch / max_epochs + math.pi
        frac = (1 + math.cos(cosv)) / 2  # smoothly oscillates in [0,1]
        k = self.k_min + frac * (self.k_max - self.k_min)
        round_k = round(k)
        if round_k == self.k_max:
            return {round_k: 1/2, round_k - 1: 1/2}
        if round_k == self.k_min:
            return {round_k: 1/2, round_k + 1: 1/2}
        return {round_k: 1/3, round_k + 1: 1/3, round_k - 1: 1/3}


class RandomScheduler(BaseScheduler):
    """k uniformly random in [k_min..k_max] each call.
    This is not a real scheduler, but a random sampling of k values.
    so we will use it to see if our schedulers are better than random sampling."""
    def _get_k_distribution(self, epoch, max_epochs):
        ks = list(range(self.k_min, self.k_max + 1))
        prob = 1/len(ks)
        return {k: prob for k in ks}


class VanillaScheduler(BaseScheduler):
    """Vanilla (non-adversarial) training baseline - returns k=0 (no adversarial examples)."""
    def __init__(self, epsilon_max: float = 0.0):
        # For vanilla training, we set k_min=k_max=0 and epsilon=0
        super().__init__(k_min=0, k_max=0, epsilon_max=epsilon_max)
        
    def _get_k_distribution(self, epoch, max_epochs):
        # Always return k=0 for vanilla training (no adversarial examples)
        return {0: 1.0}


"""
Here we define the CompositeScheduler class, which combines a K scheduler with an epsilon strategy.
The CompositeScheduler class allows us to combine different K schedulers with different epsilon strategies.
The epsilon strategies are:
  - 'constant': epsilon = epsilon_max
  - 'linear':   linear ramp epsilon_min→epsilon_max
  - 'cyclic':   cosinusoidal oscillation epsilon_min→epsilon_max→epsilon_min
  - 'random':   uniform random in [epsilon_min, epsilon_max]
  
  The CompositeScheduler class takes as input a K scheduler and an epsilon strategy, 
  and returns the epsilon value and the K distribution for the current epoch.

"""


class CompositeScheduler:
    """
    Args:
      - k_scheduler: instance of one K scheduler (Constant, Linear, Cyclic, Random)
      - epsilon_type: one of 'constant','linear','cyclic','random'
      - epsilon_min, epsilon_max: float bounds
    """
    def __init__(self, k_scheduler, epsilon_type='constant', epsilon_min=0.0, epsilon_max=0.3):
        self.k_scheduler = k_scheduler
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_type = epsilon_type

    def __call__(self, epoch: int, max_epochs: int):
        if self.epsilon_type == 'constant':
            eps = self.epsilon_max
        elif self.epsilon_type == 'linear':
            frac = epoch / max_epochs
            eps = self.epsilon_min + frac * (self.epsilon_max - self.epsilon_min)
        elif self.epsilon_type == 'cyclic':
            cosv = math.pi * epoch / max_epochs
            frac = (1 + math.cos(cosv)) / 2
            eps = self.epsilon_min + frac * (self.epsilon_max - self.epsilon_min)
        elif self.epsilon_type == 'random':
            eps = random.uniform(self.epsilon_min, self.epsilon_max)
        else:
            raise ValueError(f"Unknown epsilon_type {self.epsilon_type}")
        _, k_dist = self.k_scheduler(epoch, max_epochs)
        return eps, k_dist






"""chatgpt a fait les tests je vias devoir les verifs plus tards"""
# --- Unit tests ---
if __name__ == '__main__':
    max_epochs = 10

    """
    #sched = ConstantScheduler(0, 15)
    sched = RandomScheduler(0, 15)
    for epoch in range(max_epochs):
        print(sched._get_k_distribution(epoch, max_epochs))
    assert False
    """

    print("=== Testing K Schedulers ===")
    # 1) ConstantScheduler
    cs = ConstantScheduler(1, 2)  # Fix: provide both k_min and k_max
    expected_eps_cs = cs.epsilon_max
    expected_k_cs = 2
    for e in [0, 5, 10]:
        eps, kdist = cs(e, max_epochs)
        k_val = list(kdist.keys())[0]
        print(f"ConstScheduler epoch={e}: actual eps={eps}, expected eps={expected_eps_cs}; "
              f"actual k={k_val}, expected k={expected_k_cs}")
        assert eps == expected_eps_cs, f"epsilon mismatch at epoch {e}"
        assert k_val == expected_k_cs, f"k mismatch at epoch {e}"

    # 2) LinearScheduler
    ls = LinearScheduler(0, 6)
    print("\nLinearScheduler results:")
    for e, exp_eps, exp_k in [
        (0, ls.epsilon_max, 0),
        (5, ls.epsilon_max, 3),
        (10, ls.epsilon_max, 6)
    ]:
        eps, kdist = ls(e, max_epochs)
        k_val = list(kdist.keys())[0]
        print(f"epoch={e}: actual eps={eps}, expected eps={exp_eps}; "
              f"actual k={k_val}, expected k={exp_k}")
        assert eps == exp_eps, f"epsilon mismatch at epoch {e}"
        assert k_val == exp_k, f"k mismatch at epoch {e}"

    # 3) LinearUniformMixScheduler
    lums = LinearUniformMixScheduler(0, 8)
    e = 4
    eps, kdist = lums(e, max_epochs)
    max_k = 0 + round(e / max_epochs * (8 - 0))
    expected_keys = set(range(0, max_k + 1))
    print(f"\nLinearUniformMixScheduler epoch={e}: actual keys={set(kdist.keys())}, "
          f"expected keys={expected_keys}")
    assert set(kdist.keys()) == expected_keys, "key set mismatch"

    print("\nK tests passed!\n")

    print("=== Testing CompositeSchedulers ===")
    # A) Linear K + Linear epsilon
    comp1 = CompositeScheduler(LinearScheduler(0, 6), 'linear', 0.0, 0.3)
    for e, exp_eps, exp_k in [
        (0, 0.0, 0),
        (5, 0.15, 3),
        (10, 0.3, 6)
    ]:
        eps, kdist = comp1(e, max_epochs)
        k_val = list(kdist.keys())[0]
        print(f"Composite1 epoch={e}: actual eps={eps}, expected eps={exp_eps}; "
              f"actual k={k_val}, expected k={exp_k}")
        assert abs(eps - exp_eps) < 1e-8, f"epsilon mismatch at epoch {e}"
        assert k_val == exp_k, f"k mismatch at epoch {e}"
    print("Composite1 passed!\n")

    # B) Exponential K + Cyclic epsilon
    comp2 = CompositeScheduler(ExponentialScheduler(1, 8), 'cyclic', 0.0, 0.3)
    for e, exp_eps, exp_k in [
        (0, 0.3, 1),
        (5, 0.15, 3),
        (10, 0.0, 8)
    ]:
        eps, kdist = comp2(e, max_epochs)
        k_val = list(kdist.keys())[0]
        print(f"Composite2 epoch={e}: actual eps={eps}, expected eps={exp_eps}; "
              f"actual k={k_val}, expected k={exp_k}")
        assert abs(eps - exp_eps) < 1e-8, f"epsilon mismatch at epoch {e}"
        assert k_val == exp_k, f"k mismatch at epoch {e}"
    print("Composite2 passed!\n")

    # C) Random K + Random epsilon
    comp3 = CompositeScheduler(RandomScheduler(0, 5), 'random', 0.1, 0.2)
    e = 3
    eps, kdist = comp3(e, max_epochs)
    keys = set(kdist.keys())
    expected_keys = set(range(0, 6))
    print(f"Composite3 epoch={e}: actual eps={eps}, expected eps in [0.1,0.2]; "
          f"actual keys={keys}, expected keys={expected_keys}")
    assert 0.1 <= eps <= 0.2, "epsilon out of range"
    assert keys == expected_keys, "key set mismatch"
    assert abs(sum(kdist.values()) - 1.0) < 1e-8, "probabilities sum mismatch"
    print("Composite3 passed!\n")

    print("All tests successful!")

