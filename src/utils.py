from torch.distributions import MultivariateNormal


def detach_mvn(dist: MultivariateNormal):
    """
        detaches mean and cov of the input dist and returns a new dist
    """

    return MultivariateNormal(
        loc=dist.loc.detach(),
        scale_tril=dist.scale_tril.detach(),
    )