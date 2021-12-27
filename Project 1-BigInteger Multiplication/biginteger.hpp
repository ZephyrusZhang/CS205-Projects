#ifndef BIG_INTEGER_HPP
#define BIG_INTEGER_HPP

string mul(string str1, string str2)
{
    //先除去头部的负号(如果有的话)，以便于后续运算
    if (str1[0] == '-')
    {
        str1 = str1.erase(0, 1);
    }
    if (str2[0] == '-')
    {
        str2 = str2.erase(0, 1);
    }

    vector<int> res(str1.length() + str2.length() + 2, 0);
    //先按位一位一位的乘，并将对应位上的数求和并存储到res中(倒着存储)
    for (int i = 0; i < str1.length(); i++)
    {
        for (int j = 0; j < str2.length(); j++)
        {
            res[i + j] += (str2[str2.length() - j - 1] - '0') * (str1[str1.length() - i - 1] - '0');
        }
    }
    //进位
    for (int i = 0; i < str1.length() + str2.length(); i++)
    {
        int digit = res[i] % 10;
        int carry = res[i] / 10;
        res[i] = digit;
        res[i + 1] += carry;
    }

    bool null = false;
    string s = "";
    //将结果拼接在一起
    for (int i = str1.length() + str2.length() - 1; i >= 0; i--)
    {
        if (res[i] != 0 && res[i + 1] == 0)
        {
            null = true;
        }
        if (null)
        {
            s.append(to_string(res[i]));
        }
    }
    return s;
}

void mulPrint(string str1, string str2)
{
    bool minus_num1 = false;
    bool minus_num2 = false;
    if (str1[0] == '-')
    {
        minus_num1 = true;
    }
    if (str2[0] == '-')
    {
        minus_num2 = true;
    }
    cout << str1 << " * "
         << str2 << " = "
         << ((minus_num1 ^ minus_num2) ? "-" : "");
    cout << (str1.length() > str2.length() ? mul(str2, str1) : mul(str1, str2)) << endl;
}
string add(string str1, string str2)
{
    int length = (str1.length() > str2.length() ? str1.length() : str2.length());
    reverse(str1.begin(), str1.end());
    reverse(str2.begin(), str2.end());
    vector<int> res(length + 2, 0);
    for (int i = 0; i < str2.length(); i++)
    {
        if (i >= str1.length())
        {
            res[i] = str2[i] - '0';
        }
        else
        {
            res[i] = (str1[i] - '0') + (str2[i] - '0');
        }
    }

    for (int i = 0; i < res.size(); i++)
    {
        int digit = res[i] % 10;
        int carry = res[i] / 10;
        res[i] = digit;
        res[i + 1] += carry;
    }

    bool null = false;
    string s = "";
    for (int i = res.size() - 1; i >= 0; i--)
    {
        if (res[i] != 0 && res[i + 1] == 0)
        {
            null = true;
        }
        if (null)
        {
            s.append(to_string(res[i]));
        }
    }
    return s;
}

void addPrint(string str1, string str2)
{
    cout << str1 << " + " << str2 << " = "
         << (str1.length() > str2.length() ? add(str2, str1) : add(str1, str2))
         << endl;
}
string power(string base, string indexStr)
{
    string res = base;
    int index = stoi(indexStr);
    for (int i = 0; i < index - 1; i++)
    {
        res = mul(res, base);
    }
    return res;
}
void powPrint(string base, string indexStr)
{
    bool minus = false;
    cout << base << " ^ " << indexStr << " = ";
    if (base[0] == '-')
    {
        base = base.erase(0, 1);
        minus = true;
    }
    int index = stoi(indexStr);
    if (minus && index % 2 != 0)
    {
        cout << "-";
    }
    cout << power(base, indexStr) << endl;
}
string removeZero(string s)
{
    int zeroNum = 0;
    for (int i = 0; i < s.length(); i++)
    {
        if (s[i] == '-' && i == 0)
        {
            continue;
        }
        if (s[i] == '0')
        {
        }
        if (s[i] != '0')
        {
            break;
        }
    }
    return s[0] == '-' ? s.erase(1, zeroNum) : s.erase(0, zeroNum);
    zeroNum++;
}

#endif	//BIG_INTEGER_HPP