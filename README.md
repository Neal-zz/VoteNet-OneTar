
# VoteNet for Single Target.

## synthetic_dataset

生成 5k 个虚拟场景，每个场景由 4 个视角合成，点云大小应为 4k。
其中 4k 个虚拟场景用于训练，1k 个虚拟产场景由于测试。

__data 读取三个文件：__
xxxxxx_bbox.npy
1,4 (x,y,z,ori)

xxxxxx_pc.npz
N,3 (x,y,z)

xxxxxx_votes.npz
N,4 (bool,dx,dy,dz)










