class PositionalEncoder():
  def __init__(self):
    pass

  def build(self, input_shape):
    self.max_length = input_shape[1]
    self.d_model = input_shape[2]

  def positional_encoding(self, angle_rads):
    angle_rads[::2] = np.sin(angle_rads[::2])
    angle_rads[1::2] = np.cos(angle_rads[1::2])
    return angle_rads

  def get_angles(self, pos, i):
    return pos * (1 / ( np.power(10000, ((2*i)/self.d_model))))

  def fit(self):
    angle_rads = np.array([[self.get_angles(pos, i) for i in range(self.d_model)] for pos in range(self.max_length)])
    return self.positional_encoding(angle_rads)

  def __call__(self, inputs):
    self.build(inputs.shape)
    pos_encoding = self.fit()
    return pos_encoding + inputs

#----

class AddAndNorm():
  def __init__(self, dropout_rate=0.1):
    self.dropout_rate = dropout_rate

  def normalization(self, inputs):
    for row in range(inputs.shape[0]):
      inputs[row] = (inputs[row] - inputs[row].mean()) / inputs[row].std()
    return inputs

  def build(self, inputs, skip_connection):
    inputs = inputs + skip_connection
    return self.normalization(inputs)

  def __call__(self, inputs, skip_connection):
    output = self.build(inputs, skip_connection)
    return output

#-----

class Attention():
  def __init__(self):
    pass

  def build(self, input):
    input_shape = input.shape
    self.batch_size = input_shape[0]
    self.max_length = input_shape[1]
    self.embed_dim = input_shape[2]

  def softmax(self, vector):
    sum_of_probab = sum(np.exp(vector))
    probab_vector = np.exp(vector) / sum_of_probab
    return probab_vector

  def attention_score(self, query, key, value):
    dot_product = np.dot(query, key.transpose())
    score = np.dot(self.softmax(dot_product / np.sqrt(self.embed_dim)), value)
    return score.reshape(self.batch_size, self.max_length, self.embed_dim)

  def __call__(self, query, key, value):
    self.build(query)
    query = query.reshape(-1, query.shape[-1])
    key = key.reshape(-1, key.shape[-1])
    value = value.reshape(-1, value.shape[-1])
    return self.attention_score(query, key, value)
