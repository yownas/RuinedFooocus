def broken_torch_platforms(torch_platform, os_platform):

    if torch_platform == "xpu" and not os_platform == "Windows":
        torch_platform = "cpu"
    if torch_platform == "mps" and not os_platform == "Darwin":
        torch_platform = "cpu"
    if torch_platform == "rocm5.5": # 25.09.01 - the torch for romc5.5 is too old
        torch_platform = "cpu"

    return (torch_platform, os_platform)

