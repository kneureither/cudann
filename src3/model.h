#include <vector>
#include <cstdint>


template <typename T>
class Tensor {
public:

    Tensor() {
        this->shape = {};
        this->data_size = 0;
        this->owns_data = true;
    };

    ~Tensor() {
        if (this->owns_data) {
            delete[] data_ptr;
        }
    };

    std::vector<int> get_shape() {return this->shape;} const;
    int get_size_data() {return this->data_size;} const;


    void zeros(std::vector<int> shape) {
        if (shape.size() < 1 || shape.size() > 2) {
            throw std::invalid_argument("Tensor shape must be 1D (vector) or 2D (matrix)");
        }
        
        this->shape = shape;
        this->data_size = 1;
        for (int i = 0; i < shape.size(); i++) {
            this->data_size *= shape[i];
        }

        // allocate memory for the data
        this->data_ptr = new T[this->data_size];
        for (int i = 0; i < this->data_size; i++) {
            data_ptr[i] = T();
        }
        this->owns_data = true;
        
        if (shape.size() == 1) {
            this->view_idx.push_back(data_ptr);
        }

        if(shape.size() == 2) {
            for (int i = 0; i < shape[0]; i++) {
                this->view_idx.push_back(data_ptr+i*shape[1]*sizeof(T));
        }

    };

    void random(std::vector<int> shape) {
        this->zeros(shape);
        for (int i = 0; i < this->data_size; i++) {
            this->data_ptr[i] = T(rand() % RAND_MAX());
        }
    };

    Tensor<T> operator+(const Tensor<T> &other) {
        if (this->shape != other.shape) {
            throw std::invalid_argument("Tensor shapes must match for addition");
        }
        Tensor<T> result;
        result.zeros(this->shape);

        if (this->shape.size() == 1) {
            for (int i = 0; i < this->shape[0]; i++) {
                result[i] = this->view_idx[i] + other.view_idx[i];
            }
        } else if (this->shape.size() == 2) {
            for (int i = 0; i < this->shape[0]; i++) {
                for (int j = 0; j < this->shape[1]; j++) {
                    result[i][j] = this->view_idx[i][j] + other.view_idx[i][j];
                }
            }
        }
        return result;
    };

    Tensor<T> operator-(const Tensor<T> &other) {
        if (this->shape != other.shape) {
            throw std::invalid_argument("Tensor shapes must match for subtraction");
        }
        Tensor<T> result;
        result.zeros(this->shape);

        if (this->shape.size() == 1) {
            for (int i = 0; i < this->shape[0]; i++) {
                result[i] = this[i] - other[i];
            }
        } else if (this->shape.size() == 2) {
            for (int i = 0; i < this->shape[0]; i++) {
                for (int j = 0; j < this->shape[1]; j++) {
                    result[i][j] = this[i][j] - other[i][j];
                }
            }
        }
        return result;
    };
    
    Tensor<T>& operator=(const Tensor<T> &other) {
        this->zeros(other.shape);
        for (int i = 0; i < this->data_size; i++) {
            this->data_ptr[i] = other.data_ptr[i];
        }

        if (this->shape.size() == 1) {
            this->view_idx[0] = other.view_idx[0];
        } else if (this->shape.size() == 2) {
            for (int i = 0; i < this->shape[0]; i++) {
                this->view_idx[i] = other.view_idx[i];
            }
        }
        return *this;
    };
    
    Tensor<T> dot(const Tensor<T> &other) const;
    Tensor<T> slice(int start, int end) const;

    auto operator[](int index) {
        if (index < 0 || index >= shape[0]) {
            throw std::out_of_range("Index out of bounds");
        }
        
        if (shape.size() == 1) {
            return *(view_idx[0] + index);
        } else {
            return view_idx[index];
        }
    };

    std::string to_string() {
        std::stringstream ss;
        
        if (this->shape.size() == 1) {
            ss << "[";
            for (int i = 0; i < this->shape[0]; i++) {
                ss << this->view_idx[0][i] << ", ";
            }
            ss << "]";
        } else if (this->shape.size() == 2) {
            ss << "[";
            for (int i = 0; i < this->shape[0]; i++) {
                for (int j = 0; j < this->shape[1]; j++) {
                    ss << this->view_idx[i][j] << ", ";
                }
                ss << "],\n";
            }
            ss << "]";
        }
        return ss.str();
    };


private:
    std::vector<int> shape; 
    std::vector<T*> view_idx;
    bool owns_data = true;
    
    T* data_ptr;
    int data_size;

};

class Layer
{
public:
    Layer();
    ~Layer();

    Tensor<float> forward(Tensor<float> input);
    Tensor<float> backward(Tensor<float> input, Tensor<float> );
};

class Model
{
public:
    Model();
    ~Model();

    int predict(Tensor<float> input);


private:
    std::vector<Layer> layers;
};
