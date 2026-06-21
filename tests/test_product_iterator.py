from product_iterator import ProductIterator
import pytest

@pytest.fixture
def prod_it(request):
    return ProductIterator(a=["x","y"],b=[1,2],c=[3.14, 2.71, 1.414])

def test_view(prod_it):
    ...

def test_item_at(prod_it):
    ...

def test_yield_outer(prod_it):
    ...

def test_split(prod_it):
    ...

def test_chunk(prod_it):
    ...

def test_partition(prod_it):
    ...

