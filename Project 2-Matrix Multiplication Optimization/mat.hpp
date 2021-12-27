#ifndef MAT_HPP
#define MAT_HPP

class Matrix
{
private:
    //attributes
    float **fmat;
    double **dmat;
    unsigned int row;
    unsigned int col;
    string filename;

    /**
     * @brief 获取矩阵的行数与列数
     */
    void setSize()
    {
        ifstream ifs;
        ifs.open(filename, ios::in);
        if (!ifs.is_open())
        {
            cout << "读取矩阵 " << filename << " 失败，程序退出" << endl;
            exit(100);
        }
        string buffer;
        while (getline(ifs, buffer))
        {
            row++;
        }
        ifs.close();
        ifs.open(filename, ios::in);
        if (getline(ifs, buffer))
        {
            for (int i = 0; i < buffer.length(); i++)
            {
                if (buffer[i] == 32)
                {
                    col++;
                }
            }
            col++;
        }
        ifs.close();
    }

public:
    /**
     * @brief 构造矩阵
     * @param filename 要读取的txt文件的名称
     */
    Matrix(string filename)
    {
        row = 0;
        col = 0;
        this->filename = filename;
        setSize();
    }

    unsigned getRow()
    {
        return row;
    }

    unsigned getCol()
    {
        return col;
    }

    /**
     * @brief 将txt文件中的矩阵读取到二维数组中去
     */ 
    void prepareMat()
    {
        fmat = new float *[row];
        dmat = new double *[row];
        for (int i = 0; i < row; i++)
        {
            fmat[i] = new float[col];
            dmat[i] = new double[col];
        }

        //Initialization
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                fmat[i][j] = 0.0f;
                dmat[i][j] = 0.0;
            }
        }

        //Read file
        ifstream ifs;
        ifs.open(filename, ios::in);
        if (!ifs.is_open())
        {
            cout << "读取矩阵 " << filename << " 失败，程序退出" << endl;
            exit(100);
        }
        char buffer[32];
        unsigned int i = 0;
        unsigned int j = 0;
        while (ifs >> buffer)
        {
            if (i == row && j == col)
            {
                break;
            }
            fmat[i][j] = stof(buffer);
            dmat[i][j] = stod(buffer);
            if (++j >= col)
            {
                i++;
                j = 0;
            }
        }
        ifs.close();
    }

    /**
     * @brief float矩阵的乘法，乘法顺序为当前对象的矩阵乘以参数的矩阵
     * @param mat 另一个矩阵对象
     */  
    void fmatMul(Matrix mat)
    {
        float **res = new float *[this->row];
        for (int j = 0; j < this->row; j++)
        {
            res[j] = new float[mat.col];
        }
        //Initialization
        for (int i = 0; i < this->row; i++)
        {
            for (int j = 0; j < mat.col; j++)
            {
                res[i][j] = 0.0f;
            }
        }

        time_t start, end;
        start = clock();

        //将矩阵按照 i->j->k 的顺序进行乘法运算
        for (int i = 0; i < this->row; i++)
        {
            for (int j = 0; j < mat.col; j++)
            {
                float c_i_j = 0.0f;
                for (int k = 0; k < this->col; k++)
                {
                    c_i_j += this->fmat[i][k] * mat.fmat[k][j];
                }
                res[i][j] = c_i_j;
            }
        }

        end = clock();
        printf("%dx%d的矩阵与%dx%d的矩阵以float,按ijk的顺序相乘得出结果共耗时：%fs\n", this->row, this->col, mat.row, mat.col, (double(end - start) / CLOCKS_PER_SEC));

        //将矩阵中的每个元素都重置，为下次运算做准备
        for (int i = 0; i < this->row; i++)
        {
            for (int j = 0; j < mat.col; j++)
            {
                res[i][j] = 0.0f;
            }
        }

        start = clock();

        ////将矩阵按照 i->k->j 的顺序进行乘法运算
        for (int i = 0; i < this->row; i++)
        {
            for (int k = 0; k < this->col; k++)
            {
                float temp = this->fmat[i][k];
                for (int j = 0; j < mat.col; j++)
                {
                    res[i][j] += temp * mat.fmat[k][j];
                }
            }
        }

        end = clock();
        printf("%dx%d的矩阵与%dx%d的矩阵以float,按ikj的顺序相乘得出结果共耗时：%fs\n", this->row, this->col, mat.row, mat.col, (double(end - start) / CLOCKS_PER_SEC));

        //将结果写入txt文件中
        string ofile_name = "out-float-" + to_string(this->row) + "x" + to_string(mat.col) + ".txt";
        ofstream ofs;
        ofs.open(ofile_name, ios::out);
        for (int i = 0; i < this->row; i++)
        {
            for (int j = 0; j < mat.col; j++)
            {
                ofs << res[i][j] << " ";
            }
            ofs << endl;
        }

        //释放内存
        for (int i = 0; i < this->row; i++)
        {
            delete[] res[i];
        }
        delete[] res;
    }

    /**
     * @brief double矩阵的乘法，乘法顺序为当前对象的矩阵乘以参数的矩阵
     * @param mat 另一个矩阵对象
     */  
    void dmatMul(Matrix mat)
    {
        double **res = new double *[this->row];
        for (int j = 0; j < this->row; j++)
        {
            res[j] = new double[mat.col];
        }
        //Initialization
        for (int i = 0; i < this->row; i++)
        {
            for (int j = 0; j < mat.col; j++)
            {
                res[i][j] = 0.0;
            }
        }

        time_t start, end;
        start = clock();
        for (int i = 0; i < this->row; i++)
        {
            for (int j = 0; j < mat.col; j++)
            {
                double c_i_j = 0.0;
                for (int k = 0; k < this->col; k++)
                {
                    c_i_j += this->dmat[i][k] * mat.dmat[k][j];
                }
                res[i][j] = c_i_j;
            }
        }
        end = clock();
        printf("%dx%d的矩阵与%dx%d的矩阵以double,按ijk的顺序相乘得出结果共耗时：%fs\n", this->row, this->col, mat.row, mat.col, (double(end - start) / CLOCKS_PER_SEC));

        for (int i = 0; i < this->row; i++)
        {
            for (int j = 0; j < mat.col; j++)
            {
                res[i][j] = 0.0;
            }
        }

        start = clock();
        for (int i = 0; i < this->row; i++)
        {
            for (int k = 0; k < this->col; k++)
            {
                double temp = this->dmat[i][k];
                for (int j = 0; j < mat.col; j++)
                {
                    res[i][j] += temp * mat.dmat[k][j];
                }
            }
        }
        end = clock();
        printf("%dx%d的矩阵与%dx%d的矩阵以double,按ikj的顺序相乘得出结果共耗时：%fs\n", this->row, this->col, mat.row, mat.col, (double(end - start) / CLOCKS_PER_SEC));

        string ofile_name = "out-double-" + to_string(this->row) + "x" + to_string(mat.col) + ".txt";
        ofstream ofs;
        ofs.open(ofile_name, ios::out);
        for (int i = 0; i < this->row; i++)
        {
            for (int j = 0; j < mat.col; j++)
            {
                ofs << res[i][j] << " ";
            }
            ofs << endl;
        }

        for (int i = 0; i < this->row; i++)
        {
            delete[] res[i];
        }
        delete[] res;
    }

    /**
     * @brief 将矩阵打印到命令行上
     */ 
    void printMat()
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                printf("%5.1f ", dmat[i][j]);
            }
            cout << endl;
        }
    }
};

#endif //MAT_HPP