
H_MAZE = \
        "####################\\"+\
        "#OOOOOOOOOOOOOOOOOO#\\"+\
        "#OO#######OOOOOOOOO#\\"+\
        "#OOOOOOOOOOO#####OO#\\"+\
        "#OOOOOOOOOOOOOOOOOO#\\"+\
        "####################"

M_MAZE_V7= \
        "####################\\"+\
        "#OOOOOOOO#OOOOOOOOO#\\"+\
        "#OOO############OOO#\\"+\
        "#O#OO#####O####OO#O#\\"+\
        "#O##OO####O###OO##O#\\"+\
        "#O###OO###O##OO###O#\\"+\
        "#O####OO##O#OO####O#\\"+\
        "#O#####OO#OOO#####O#\\"+\
        "#O######OOOO######O#\\"+\
        "###OOOOOOOOO#OOOOOO#\\"+\
        "#O######OOOO########\\"+\
        "#O#####OO##OO#####O#\\"+\
        "#O####OO####OO####O#\\"+\
        "#O###OO##O###OO###O#\\"+\
        "#O##OO###O####OO##O#\\"+\
        "#O#OO####O#####OO#O#\\"+\
        "#OOO#####O######OOO#\\"+\
        "#OOOOOOOOOOOOOO#OOO#\\"+\
        "####################"

M_MAZE2_V1 = \
        "####################\\"+\
        "#OOOOOOO#OOOOOOOOOO#\\"+\
        "#OOOOOOO#OOOO###OOO#\\"+\
        "#OOOOOO###OOOO#OOOO#\\"+\
        "#OOOOOOOOOOOOO#OOOO#\\"+\
        "#OOOOOO###OOOO#OOOO#\\"+\
        "#OOOOOOO#OOOOO#OOOO#\\"+\
        "#OOOOOOO#OOOOO#OOOO#\\"+\
        "#OOOOOOO#OOOOO#OOOO#\\"+\
        "#OOOO#O#############\\"+\
        "######O#O#OOOOOOOOO#\\"+\
        "#OOOO#O#O#OOOOOOOOO#\\"+\
        "#OOOOOOOO#OOOOO#O#O#\\"+\
        "#OOOOOOOO#OO####O###\\"+\
        "#OOOOOOOO#OO#OO#O#O#\\"+\
        "#OO#OOOOO#OO#OOOOOO#\\"+\
        "#OO#######OO#OOOOOO#\\"+\
        "#OO#OOOOOOOO#OOOOOO#\\"+\
        "#OOOOOOOOOOO#OOOOOO#\\"+\
        "####################"


maze_name_space= {
    'h_maze' : H_MAZE,
    'm_maze1' : M_MAZE_V7,
    'm_maze2' : M_MAZE2_V1,
}

LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OGO#\\"+\
        "############"

LARGE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OGO#\\"+\
        "##O###O#O#O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "#O##O#OO##O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O##O#O#O###\\"+\
        "#OOOO#OOOOO#\\"+\
        "############"

MEDIUM_MAZE = \
        '########\\'+\
        '#OO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#OOO#OG#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOG#\\'+\
        '#O#O##O#\\'+\
        '#OOOO#O#\\'+\
        '###OO###\\'+\
        '#OOOOOO#\\'+\
        '#OO##OO#\\'+\
        "########"

SMALL_MAZE = \
        "######\\"+\
        "#OOOO#\\"+\
        "#O##O#\\"+\
        "#OOOO#\\"+\
        "######"

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#######"

HARD_EXP_MAZE = \
        "#####################\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#OOO###OOO#OOO###OOO#\\"+\
        "#OOOOOOOOO#OOOOOOOOO#\\"+\
        "#OOO###OOO#OOO###OOO#\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#OOOO#OOO###OOO#OOOO#\\"+\
        "#OOOO#OOOOOOOOO#OOOO#\\"+\
        "#OOOO#OOO###OOO#OOOO#\\"+\
        "#OOOO#OOOO#OOOO#OOOG#\\"+\
        "#####################"


HARD_EXP_MAZE_V2 = \
        "#####################\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#OOO###OOO#OOO###OOO#\\"+\
        "#OOOOOOOOO#OOOOOOOOO#\\"+\
        "#OOO###OOO#OOO###OOO#\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#OOOO#OOO###OOO#OOOO#\\"+\
        "#OOOO#OOOOOOOOO#OOOO#\\"+\
        "#OOOO#OOO###OOO#OOOO#\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#################OG##\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#OOOO#OOO###OOO#OOOO#\\"+\
        "#OOOO#OOOOOOOOO#OOOO#\\"+\
        "#OOOO#OOO###OOO#OOOO#\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#OOO###OOO#OOO###OOO#\\"+\
        "#OOOOOOOOO#OOOOOOOOO#\\"+\
        "#OOO###OOO#OOO###OOO#\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#####################"