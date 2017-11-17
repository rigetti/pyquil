def wait_for_job(res, ping_time=0.5):
    raise DeprecationWarning(
        "The wait_for_job function is now deprecated. See https://go.rigetti.com/connections for more info.")


class JobResult(object):
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning("JobResult has been deprecated in favor of pyquil.api.Job")
