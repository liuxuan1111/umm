#include <preliminaries.hpp>
#include <ublas.hpp>
#include <iostream>
#include <Eigenvalues>

namespace umm
{
    EigenMatrix to_eigenmatrix(const matrix& m)
    {
        EigenMatrix m_out(m.rows(), m.cols());
        for (size_t i = 0; i < m.rows(); ++i)
        {
            for (size_t j = 0; j < m.cols(); ++j)
                m_out(i, j) = m(i, j);
        }
        return m_out;
    }

    matrix to_matrix(const EigenMatrix& m)
    {
        matrix m_out(m.rows(), m.cols());
        for (size_t i = 0; i < m_out.rows(); ++i)
        {
            for (size_t j = 0; j < m_out.cols(); ++j)
                m_out(i, j) = m(i, j);
        }
        return m_out;
    }

    matrix::matrix(const double c, const size_t rows, const size_t cols) : m_rows(rows), m_cols(cols), m_is_diagonal(false)
    {
        m_data = std::vector<double>(m_rows * m_cols, c);
        if (m_rows == m_cols)
            m_is_diagonal = true;
    }

    matrix::matrix(const size_t rows, const size_t cols) : m_rows(rows), m_cols(cols), m_is_diagonal(false)
    {
        m_data = std::vector<double>(m_rows * m_cols);
    }

    matrix::matrix(const std::vector<double>& data) : m_rows(data.size()), m_cols(data.size()), m_is_diagonal(true)
    {
        m_data = std::vector<double>(m_rows * m_cols, 0.0);
        for (size_t i = 0; i < m_rows; ++i)
            m_data[i * m_cols + i] = data[i];
    }

    matrix::matrix(const size_t dim) : m_rows(dim), m_cols(dim), m_is_diagonal(true)
    {
        m_data = std::vector<double>(m_rows * m_cols, 0.0);
    }

    matrix matrix::operator-() const
    {
        matrix m_out(m_rows, m_cols);
        for (size_t i = 0; i < m_rows; ++i)
        {
            for (size_t j = 0; j < m_cols; ++j)
                m_out(i, j) = -(*this)(i, j);
        }
        return m_out;
    }

    matrix matrix::transpose() const
    {
        matrix m_out(m_cols, m_rows);
        for (size_t i = 0; i < m_rows; ++i)
        {
            for (size_t j = 0; j < m_cols; ++j)
                m_out(j, i) = (*this)(i, j);
        }
        return m_out;
    }

    matrix matrix::inverse() const
    {
        return to_matrix(to_eigenmatrix(*this).inverse());
    }

    double matrix::trace() const
    {
        const size_t n = std::min(m_rows, m_cols);
        double tr = 0.0;
        for (size_t i = 0; i < n; ++i)
            tr += (*this)(i, i);
        return tr;
    }

    eigen_solver::eigen_solver(const matrix& m)
    {
        Eigen::SelfAdjointEigenSolver<EigenMatrix> es(to_eigenmatrix(m));
        m_eigen_values = to_matrix(es.eigenvalues().asDiagonal());
        m_eigen_values.set_is_diagonal(true);
        m_eigen_vectors = to_matrix(es.eigenvectors());
    }

    matrix operator*(const matrix& m1, const matrix& m2)
    {
        VERIFY(m1.cols() == m2.rows(), "incompatible size.");

        matrix m_out(m1.rows(), m2.cols());

        if (m1.is_diagonal() && m2.is_diagonal())
        {
            for (size_t i = 0; i < m1.rows(); ++i)
            {
                m_out(i, i) = m1(i, i) * m2(i, i);
            }
            m_out.set_is_diagonal(true);
            return m_out;
        }

        if (m1.is_diagonal() && !m2.is_diagonal())
        {
            for (size_t i = 0; i < m1.rows(); ++i)
            {
                for (size_t j = 0; j < m2.cols(); ++j)
                    m_out(i, j) = m1(i, i) * m2(i, j);
            }
            return m_out;
        }

        if (!m1.is_diagonal() && m2.is_diagonal())
        {
            for (size_t i = 0; i < m1.rows(); ++i)
            {
                for (size_t j = 0; j < m2.cols(); ++j)
                    m_out(i, j) = m1(i, j) * m2(j, j);
            }
            return m_out;
        }

        for (size_t i = 0; i < m1.rows(); ++i)
        {
            for (size_t j = 0; j < m2.cols(); ++j)
            {
                m_out(i, j) = 0.0;
                for (size_t k = 0; k < m1.cols(); ++k)
                    m_out(i, j) += m1(i, k) * m2(k, j);
            }
        }
        return m_out;
    }

    matrix operator+(const matrix& m1, const matrix& m2)
    {
        VERIFY(m1.rows() == m2.rows() && m1.cols() == m2.cols(), "incompatible size.");

        matrix m_out(m1.rows(), m1.cols());
        for (size_t i = 0; i < m1.rows(); ++i)
        {
            for (size_t j = 0; j < m1.cols(); ++j)
                m_out(i, j) = m1(i, j) + m2(i, j);
        }
        return m_out;
    }

    matrix operator-(const matrix& m1, const matrix& m2)
    {
        VERIFY(m1.rows() == m2.rows() && m1.cols() == m2.cols(), "incompatible size.");

        matrix m_out(m1.rows(), m1.cols());
        for (size_t i = 0; i < m1.rows(); ++i)
        {
            for (size_t j = 0; j < m1.cols(); ++j)
                m_out(i, j) = m1(i, j) - m2(i, j);
        }
        return m_out;
    }

    matrix operator*(const double c, const matrix& m)
    {
        matrix m_out(m.rows(), m.cols());
        for (size_t i = 0; i < m.rows(); ++i)
        {
            for (size_t j = 0; j < m.cols(); ++j)
                m_out(i, j) = c * m(i, j);
        }
        return m_out;
    }

    matrix operator*(const matrix& m, const double c)
    {
        return c * m;
    }

    matrix operator+(const double c, const matrix& m)
    {
        matrix m_out(m.rows(), m.cols());
        for (size_t i = 0; i < m.rows(); ++i)
        {
            for (size_t j = 0; j < m.cols(); ++j)
                m_out(i, j) = c + m(i, j);
        }
        return m_out;
    }

    matrix operator+(const matrix& m, const double c)
    {
        return c + m;
    }

    matrix operator-(const double c, const matrix& m)
    {
        matrix m_out(m.rows(), m.cols());
        for (size_t i = 0; i < m.rows(); ++i)
        {
            for (size_t j = 0; j < m.cols(); ++j)
                m_out(i, j) = c - m(i, j);
        }
        return m_out;
    }

    matrix operator-(const matrix& m, const double c)
    {
        return m + (-c);
    }

    matrix operator/(const matrix& m, const double c)
    {
        return (1 / c) * m;
    }

    matrix odot(const matrix& m1, const matrix& m2)
    {
        VERIFY(m1.rows() == m2.rows() && m1.cols() == m2.cols(), "incompatible size.");
        matrix m_out(m1.rows(), m1.cols());
        for (size_t i = 0; i < m1.rows(); ++i)
        {
            for (size_t j = 0; j < m1.cols(); ++j)
                m_out(i, j) = m1(i, j) * m2(i, j);
        }
        return m_out;
    }

    matrix odot(const matrix& m1, const matrix& m2, const matrix& m3)
    {
        return odot(odot(m1, m2), m3);
    }

    matrix odiv(const matrix& m1, const matrix& m2)
    {
        VERIFY(m1.rows() == m2.rows() && m1.cols() == m2.cols(), "incompatible size.");
        matrix m_out(m1.rows(), m1.cols());
        for (size_t i = 0; i < m1.rows(); ++i)
        {
            for (size_t j = 0; j < m1.cols(); ++j)
                m_out(i, j) = m1(i, j) / m2(i, j);
        }
        return m_out;
    }

    matrix diagonal(const matrix& v)
    {
        VERIFY(v.rows() == 1 || v.cols() == 1, "input matrix must have just one row or just one column.");
        return matrix(v.data());
    }

    matrix symmetric(const matrix& m)
    {
        return 0.5 * (m + m.transpose());
    }

    matrix sqrtm(const matrix& m)
    {
        eigen_solver es(m);
        matrix lamb = es.eigen_values();
        const matrix Q = es.eigen_vectors();
        for (size_t i = 0; i < m.rows(); ++i)
        {
            lamb(i, i) = std::sqrt(std::abs(lamb(i, i)));
        }
        return Q * lamb * Q.transpose();
    }

    std::ostream& operator<<(std::ostream& os, const matrix& m)
    {
        os << "[";
        for (size_t i = 0; i < m.rows(); ++i)
        {
            os << (i == 0 ? "[" : ", [");
            for (size_t j = 0; j < m.cols(); ++j)
                os << (j == 0 ? "" : ", ") << m(i, j);
            os << "]";
        }
        os << "]";
        return os;
    }
}
