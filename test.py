from mmseg.utils import collect_env

# log env info
env_info_dict = collect_env()
env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
# dash_line = '-' * 60 + '\n'
print(env_info)
#    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
#                dash_line)


