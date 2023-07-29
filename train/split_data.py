import os
import splitfolders
import shutil

input_imgfloder = "C:\\Users\\XLiang\\Desktop\\shixi\\images"
input_labelfloder = "C:\\Users\\XLiang\\Desktop\\shixi\\labels"

output_imgfloder = "C:\\Users\\XLiang\\Desktop\\shixi\\train\\img_oupt"
output_labelfloder = "C:\\Users\\XLiang\\Desktop\\shixi\\train\\label_oupt"



def main(input_dir,output_dir):
    splitfolders.ratio(input_dir, output_dir, seed=42, ratio=(0.8, 0.1, 0.1)) # 划分好了img 的训练集和 测试集
    # splitfolders.ratio(input_dir, output_dir, seed=42, ratio=(0.8, 0.1, 0.1)) # 划分好了img 的训练集和 测试集
if __name__=='__main__':
    main(input_imgfloder,output_imgfloder)
    main(input_labelfloder,output_labelfloder)

# destination_file =   "C:\\Users\\XLiang\\Desktop\\shixi\\jl"
# # 文件移动命令
# def mv(src_path,goal_path): # 划分结果的路径
#     # # 当前路径
#     # if os.path.exists(src_path):    #如何是路径
#     #     # cur_path = os.getcwd()
#     #     #递归到文件
#     #     while os.path.isfile(path) == False:  # 不是文件 
#     #             os.chdir(path)
#     for root,dirs,files in os.walk(src_path): # root当前遍历的目录路径、dirs当前目录下的子目录列表和files当前目录下的文件列表
#         # 移动完所有文件后 在去下一个目录移动文件
#         # for dir in dirs:
#         for dir in dirs:
#             if dir == 'test': # 递归回退的时候 说明test目录下的文件移动完毕
#                 pass            # 修改移动目的文件的目录
#             # for file in files:
#             #     pass
#             #     shutil.move(root,destination_file) # test/cone ---->dest
#             # if dir == 'train':
#             #     shutil.move(root,destination_file)
# mv(output_labelfloder,destination_file)




