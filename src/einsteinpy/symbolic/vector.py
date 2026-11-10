from einsteinpy.symbolic.tensor import BaseRelativityTensor


class GenericVector(BaseRelativityTensor):
    """
    Class to represent a vector in arbitrary space-time symbolically

    """

    def __init__(self, arr, syms, config="u", parent_metric=None, name="GenericVector"):
        """
        Constructor and Initializer

        Parameters
        ----------
        arr : ~sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray
            Sympy Array containing Sympy Expressions
        syms : tuple or list
            Tuple of crucial symbols denoting time-axis, 1st, 2nd, and 3rd axis (t,x1,x2,x3)
        config : str
            Configuration of contravariant and covariant indices in tensor. 'u' for upper and 'l' for lower indices. Defaults to 'u'.
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the Generic Vector.
            Defaults to None.
        name : str
            Name of the Vector. Defaults to "GenericVector".

        Raises
        ------
        ValueError
            config has more than 1 index
        ValueError
            Dimension should be equal to 1

        """
        super(GenericVector, self).__init__(
            arr=arr, syms=syms, config=config, parent_metric=parent_metric, name=name
        )
        if self.arr.rank() == 1:
            self._order = 1
            if not len(config) == self._order:
                raise ValueError("config should be of length {}".format(self._order))
        else:
            raise ValueError("Dimension should be equal to 1")

    def change_config(self, newconfig="u", metric=None):
        """
        Changes the index configuration(contravariant/covariant)

        Parameters
        ----------
        newconfig : str
            Specify the new configuration. Defaults to 'u'
        metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Parent metric tensor for changing indices.
            Already assumes the value of the metric tensor from which it was initialized if passed with None.
            Defaults to None.

        Returns
        -------
        ~einsteinpy.symbolic.vector.GenericVector
            New tensor with new configuration.

        Raises
        ------
        Exception
            Raised when a parent metric could not be found.

        """
        return super().change_config(newconfig=newconfig, metric=metric)
