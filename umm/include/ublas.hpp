#pragma once
#include <vector>
#include <Eigenvalues>
#include <Core>

namespace umm
{
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix; // use Eigen for matrix decompositions 

    class matrix
    {
    public:
        matrix() : m_data(std::vector<double>(0)), m_rows(0), m_cols(0), m_is_diagonal(false) {}
        matrix(const std::vector<double>& data, const size_t rows, const size_t cols) : m_data(data), m_rows(rows), m_cols(cols), m_is_diagonal(false) {}
        matrix(const matrix& m) : m_data(m.data()), m_rows(m.rows()), m_cols(m.cols()), m_is_diagonal(m.is_diagonal()) {}
        matrix(const double c, const size_t rows, const size_t cols);
        matrix(const size_t rows, const size_t cols);

        /* constructors for diagonal matrices.construction of vectors should always by the above constructors */
        matrix(const std::vector<double>& data);  
        matrix(const size_t dim);

        size_t rows() const { return m_rows; }
        size_t cols() const { return m_cols; }
        const std::vector<double>& data() const { return m_data; }
        bool is_diagonal() const { return m_is_diagonal; }
        void set_is_diagonal(const bool is_diagonal) { m_is_diagonal = is_diagonal; }
        double operator()(const size_t i, const size_t j) const { return m_data[i * m_cols + j]; }
        double& operator()(const size_t i, const size_t j) { return m_data[i * m_cols + j]; }
        matrix operator-() const;
        matrix transpose() const;
        matrix inverse() const;
        double trace() const;

    private:
        std::vector<double> m_data;
        size_t m_rows;
        size_t m_cols;
        bool m_is_diagonal;
    };

    class eigen_solver
    {
    public:
        eigen_solver(const matrix& m);
        matrix eigen_values() const { return m_eigen_values; }
        matrix eigen_vectors() const { return m_eigen_vectors; }

    private:
        matrix m_eigen_values;  // diagonal matrix with eigen values as diagonal elements
        matrix m_eigen_vectors;
    };

    EigenMatrix to_eigenmatrix(const matrix& m);
    
    matrix to_matrix(const EigenMatrix& m);

    matrix operator*(const matrix& m1, const matrix& m2);
    
    matrix operator+(const matrix& m1, const matrix& m2);
    
    matrix operator-(const matrix& m1, const matrix& m2);

    matrix operator*(const double c, const matrix& m);

    matrix operator*(const matrix& m, const double c);

    matrix operator+(const double c, const matrix& m);

    matrix operator+(const matrix& m, const double c);

    matrix operator-(const double c, const matrix& m);

    matrix operator-(const matrix& m, const double c);

    matrix operator/(const matrix& m, const double c);

    /* element-wise multiplication */
    matrix odot(const matrix& m1, const matrix& m2);

    /* element-wise multiplication */
    matrix odot(const matrix& m1, const matrix& m2, const matrix& m3);

    /* element-wise division */
    matrix odiv(const matrix& m1, const matrix& m2);

    /* construct diagonal matrix with v as diagonal */
    matrix diagonal(const matrix& v);

    matrix symmetric(const matrix& m);

    matrix sqrtm(const matrix& m);

    std::ostream& operator<<(std::ostream& os, const matrix& m);

}
