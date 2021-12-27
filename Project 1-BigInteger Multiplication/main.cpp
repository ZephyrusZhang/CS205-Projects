#include <iostream>
#include "biginteger.hpp"

using namespace std;

int main(int argc, char **argv)
{
    bool flag = false;

    string str1;
    string str2;
    if (argc > 1)
    {
        str1 = argv[1];
        str2 = argv[2];
        goto FLAG;
    }

    while (true)
    {
        cout << "Please input two integers" << endl;
        cin >> str1;
        if (str1.compare("quit") == 0)
        {
            exit(1000);
        }
        cin >> str2;

    FLAG:

        bool isNum = true;
        for (int i = 1; i < str1.length(); i++)
        {
            if (!isdigit(str1[i]) && (str1[0] == '-' || isdigit(str1[0])))
            {
                isNum = false;
                break;
            }
        }

        //检查输入是否合法
        for (int i = 1; i < str2.length(); i++)
        {
            if (!isdigit(str2[i]) && (str2[0] == '-' || isdigit(str2[0])))
            {
                isNum = false;
                break;
            }
        }
        if (!isNum)
        {
            cout << "Invalid input. Try again." << endl;
            continue;
        }

        //去掉前面无意义的0
        str1 = removeZero(str1);
        str2 = removeZero(str2);

        //进行乘，加，指数运算并打印出结果(加法只支持正整数加法) mulPrint(str1, str2);
        mulPrint(str1, str2);
        addPrint(str1, str2);
        powPrint(str1, str2);
    }
}