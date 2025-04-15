"""
Here we define some adversarial examples schedulers.

These schedulers are classes with a k_min and k_max (the min and max k they can return).
As functions, they take as input the current epoch and total number of epochs,
and return the probability distribution of each k value for the current epoch as a dictionnary. 

k is the number of PGD attack iterations, so together with epsilon (perturbation size)
they define the difficulty of the adversarial example.

They should all have the same signature, since they should be usable
interchangably in the training functions (we'll give them as input to 
training functions).

! Maybe a better signature could be used. We could include epsilon_max in the signature for instance ! (if we want to schedule epsilon as well)

TODO: Add some more schedulers
"""

class ConstantScheduler:
    """
    The most common scheduler. Always use a fixed k.
    """
    def __init__(self, k_min, k_max):
        assert k_min == k_max, "ConstantScheduler requires k_min == k_max"
        self.k_min = k_min
        self.k_max = k_max

    def __call__(self, epoch, max_epochs):
        """Always use a fixed k."""
        return {self.k_min: 1.0}


class LinearScheduler:
    """
    Start from k_min and linearly increase up to k_max
    """
    def __init__(self, k_min, k_max):
        self.k_min = k_min
        self.k_max = k_max

    def __call__(self, epoch, max_epochs):
        """Linearly increase k from k_min to k_max over training."""
        epoch_ratio = epoch / max_epochs
        current_k = self.k_min + round(epoch_ratio * (self.k_max - self.k_min))
        return {current_k: 1.0}


class LinearUniformMixScheduler:

    """
    The scheduler advocated by the CAT paper.

    k linearly increases from k_min (0 in the CAT paper) to k_max during the epochs.
    However, instead of only returning a proportion 1 of the chosen k (current_k), we return a uniform
    proportion of k_min, k_min + 1, ... ,current_k

    (We should check the paper to make sure that this really is correct).
    """

    def __init__(self, k_min, k_max):
        self.k_min = k_min
        self.k_max = k_max

    def __call__(self, epoch, max_epochs):
        """
        Linearly increase k_max for the current epoch,
        but return a uniform distribution over k_min, ..., current_k.
        """
        epoch_ratio = epoch / max_epochs
        max_k_for_current_epoch = self.k_min + round(epoch_ratio * (self.k_max - self.k_min))
        number_of_k_values = max_k_for_current_epoch - self.k_min + 1

        proportions = {}
        for k in range(self.k_min, max_k_for_current_epoch + 1):
            proportions[k] = 1.0 / number_of_k_values
        
        return proportions


"""
Small check to make sure schedulers behave as expected
"""
if __name__ == "__main__":

    print("constant scheduler | k_min = 2, k_max = 2, epoch = 4, max_epochs = 10")
    print("Expecting {2: 1.0}")
    scheduler = ConstantScheduler(2, 2)
    print(scheduler(8, 10))
    print("")

    print("linear scheduler | k_min = 0, k_max = 6, epoch = 2, max_epochs = 3")
    print("Expecting {4: 1.0}")
    scheduler = LinearScheduler(0, 6)
    print(scheduler(2, 3))
    print("")

    print("linear uniform mix scheduler | k_min = 0, k_max = 10, epoch = 8, max_epochs = 10")
    print("Expecting {0: 1/9, ...8: 1/9}")
    scheduler = LinearUniformMixScheduler(0, 10)
    print(scheduler(8, 10))
    print("")

