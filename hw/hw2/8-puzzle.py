import operator
import sys
 
import numpy as np
 
 
class statusObject:
    def __init__(self):
        # 当前状态的序列
        self.array = []
        # 当前状态的估价函数值
        self.Fn = 0
        # cameFrom表示该状态由上一步由何种operation得到
        # 目的是为了过滤 【死循环】
        # 0表示初始无状态 1表示up 2表示down 3表示left 4表示right
        self.cameFrom = 0
        # 第一次生成该节点时在图中的深度 计算估价函数使用
        self.Dn = 0
        self.Father = statusObject
 
 
def selectOperation(i, cameFrom):
    # @SCY164759920
    # 根据下标和cameFromReverse来选择返回可选择的操作
    selectd = []
    if (i >= 3 and i <= 8 and cameFrom != 2):  # up操作
        selectd.append(1)
    if (i >= 0 and i <= 5 and cameFrom != 1):  # down操作
        selectd.append(2)
    if (i == 1 or i == 2 or i == 4 or i == 5 or i == 7 or i == 8):  # left操作
        if (cameFrom != 4):
            selectd.append(3)
    if (i == 0 or i == 1 or i == 3 or i == 4 or i == 6 or i == 7):  # right操作
        if (cameFrom != 3):
            selectd.append(4)
    return selectd
 
 
def up(i):
    return i - 3
 
 
def down(i):
    return i + 3
 
 
def left(i):
    return i - 1
 
 
def right(i):
    return i + 1
 
def setArrayByOperation(oldIndex, array, operation):
    # i为操作下标
    # 根据operation生成新状态
    if (operation == 1):  # up
        newIndex = up(oldIndex)  # 得到交换的下标
    if (operation == 2):  # down
        newIndex = down(oldIndex)
    if (operation == 3):  # left
        newIndex = left(oldIndex)
    if (operation == 4):  # right
        newIndex = right(oldIndex)
    # 对调元素的值
    temp = array[newIndex]
    array[newIndex] = array[oldIndex]
    array[oldIndex] = temp
    return array
 
 
def countNotInPosition(current, end):  # 判断不在最终位置的元素个数
    count = 0  # 统计个数
    current = np.array(current)
    end = np.array(end)
    for index, item in enumerate(current):
        if ((item != end[index]) and item != 0):
            count = count + 1
    return count
 
 
def computedLengthtoEndArray(value, current, end):  # 两元素的下标之差并去绝对值
    def getX(index):  # 获取当前index在第几行
        if 0 <= index <= 2:
            return 0
        if 3 <= index <= 5:
            return 1
        if 6 <= index <= 8:
            return 2
 
    def getY(index):  # 获取当前index在第几列
        if index % 3 == 0:
            return 0
        elif (index + 1) % 3 == 0:
            return 2
        else:
            return 1
 
    currentIndex = current.index(value)  # 获取当前下标
    currentX = getX(currentIndex)
    currentY = getY(currentIndex)
    endIndex = end.index(value)  # 获取终止下标
    endX = getX(endIndex)
    endY = getY(endIndex)
    length = abs(endX - currentX) + abs(endY - currentY)
    return length
 
def countTotalLength(current, end):
    # 根据current和end计算current每个棋子与目标位置之间的距离和【除0】
    count = 0
    for item in current:
        if item != 0:
            count = count + computedLengthtoEndArray(item, current, end)
    return count
 
def printArray(array):  # 控制打印格式
    print(str(array[0:3]) + '\n' + str(array[3:6]) + '\n' + str(array[6:9]) + '\n')
 
def getReverseNum(array):  # 得到指定数组的逆序数 包括0
    count = 0
    for i in range(len(array)):
        for j in range(i + 1, len(array)):
            if array[i] > array[j]:
                count = count + 1
    return count
 
 
openList = []  # open表  存放实例对象
closedList = []  # closed表
endArray = [2,1,3,4,5,6,7,8,0]  # 最终状态
countDn = 0  # 执行的次数
 
initObject = statusObject()  # 初始化状态
# initObject.array = [2, 8, 3, 1, 6, 4, 7, 0, 5]
initObject.array = [2, 8, 3, 1, 6, 4, 7, 0, 5]
# initObject.array = [2, 1, 6, 4, 0, 8, 7, 5, 3]
initObject.Fn = countDn + 1*countNotInPosition(initObject.array, endArray)
# initObject.Fn = countDn + countTotalLength(initObject.array, endArray)
openList.append(initObject)
zeroIndex = openList[0].array.index(0)
# 先做逆序奇偶性判断  0位置不算
initRev = getReverseNum(initObject.array) - zeroIndex  # 起始序列的逆序数
print("起始序列逆序数", initRev)
endRev = getReverseNum(endArray) - endArray.index(0)  # 终止序列的逆序数
print("终止序列逆序数", endRev)
res = countTotalLength(initObject.array, endArray)
# print("距离之和为", res)
# @SCY164759920
# 若两逆序数的奇偶性不同，则该情况无解
 
if((initRev%2==0 and endRev%2==0) or (initRev%2!=0 and endRev%2!=0)):
    finalFlag = 0
    while(1):
        # 判断是否为end状态
        if(operator.eq(openList[0].array,endArray)):
            # 更新表，并退出
            deep = openList[0].Dn
            finalFlag = finalFlag +1
            closedList.append(openList[0])
            endList = []
            del openList[0]
            if(finalFlag == 1):
                father = closedList[-1].Father
                endList.append(endArray)
                print("最终状态为:")
                printArray(endArray)
                while(father.Dn >=1):
                    endList.append(father.array)
                    father = father.Father
                endList.append(initObject.array)
                print("【变换成功,共需要" + str(deep) +"次变换】")
                for item in reversed(endList):
                    printArray(item)
                sys.exit()
        else:
            countDn = countDn + 1
            # 找到选中的状态0下标
            zeroIndex = openList[0].array.index(0)
            # 获得该位置可select的operation
            operation = selectOperation(zeroIndex, openList[0].cameFrom)
            # print("0的下标", zeroIndex)
            # print("cameFrom的值", openList[0].cameFrom)
            # print("可进行的操作",operation)
            # # print("深度",openList[0].Dn)
            # print("选中的数组:")
            # printArray(openList[0].array)
            # 根据可选择的操作算出对应的序列
            tempStatusList = []
            for opeNum in operation:
                # 根据操作码返回改变后的数组
                copyArray = openList[0].array.copy()
                newArray = setArrayByOperation(zeroIndex, copyArray, opeNum)
                newStatusObj = statusObject()  # 构造新对象插入open表
                newStatusObj.array = newArray
                newStatusObj.Dn = openList[0].Dn + 1 # 更新dn 再计算fn
                newFn = newStatusObj.Dn + 1*countNotInPosition(newArray, endArray)
                # newFn = newStatusObj.Dn + countTotalLength(newArray, endArray)
                newStatusObj.Fn = newFn
                newStatusObj.cameFrom = opeNum
                newStatusObj.Father = openList[0]
                tempStatusList.append(newStatusObj)
            # 将操作后的tempStatusList按Fn的大小排序
            tempStatusList.sort(key=lambda t: t.Fn)
            # 更新closed表
            closedList.append(openList[0])
            # 更新open表
            del openList[0]
            for item in tempStatusList:
                openList.append(item)
            # 根据Fn将open表进行排序
            openList.sort(key=lambda t: t.Fn)
            # print("第"+str(countDn) +"次的结果:")
            # print("open表")
            # for item in openList:
            #     print("Fn" + str(item.Fn))
            #     print("操作" + str(item.cameFrom))
            #     print("深度"+str(item.Dn))
            #     printArray(item.array)
            #      @SCY164759920
            # print("closed表")
            # for item2 in closedList:
            #     print("Fn" + str(item2.Fn))
            #     print("操作" + str(item2.cameFrom))
            #     print("深度" + str(item2.Dn))
            #     printArray(item2.array)
            # print("==================分割线======================")
else:
    print("该种情况无解")