from sglang.srt.entrypoints.http_server import launch_server as srt_launch_server


def launch_server(*args, **kwargs):
    print("Launching TGLang inference server...")

    return srt_launch_server(*args, **kwargs)
