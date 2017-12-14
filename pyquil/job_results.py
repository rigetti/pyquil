def wait_for_job(res, ping_time=0.5):
    raise DeprecationWarning("""
The wait_for_job function has been moved inside the QVMConnection or
QPUConnection object. For instance:
    job = qvm.wait_for_job(job_id)
    print(job.result())

See https://go.rigetti.com/connections for more info.""")


class JobResult(object):
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning("JobResult has been deprecated in favor of pyquil.api.Job")
