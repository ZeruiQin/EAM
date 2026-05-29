# 创建本地目录
import os

# local_apk_path = '/mnt/d/project/LLM_project/GUI-explorer-main/exploration_output/com.google.android.dialer/com.google.android.dialer.apk'
#
# os.makedirs(os.path.dirname(local_apk_path), exist_ok=True)
#
# # 转换WSL路径为Windows路径
# if local_apk_path.startswith('/mnt/'):
#     drive = local_apk_path.split('/')[2].upper()
#     windows_path = local_apk_path.replace(f'/mnt/{drive.lower()}', f'{drive}:').replace('/', '\\')
# else:
#     windows_path = local_apk_path
#
# print(windows_path)
print(os.popen("ip route show | grep -i default | awk '{ print $3}'").read().strip())