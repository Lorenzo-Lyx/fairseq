#@User: This module implements the construction and encoding of a Huffman Tree. 
#       It need read encoding information from a specific file format and also supports writing encoded information in a specific format.
#           symbol<sep>count\n
#           symbol<sep>count\n
#       The HuffmanCoder is capable of efficiently writing encoding information as byte data.
#       For examples: 
#           @Code{
#               builder = HuffmanCodeBuilder.from_file(file_path)
#               coder = builder.build_code() #Encode or decode
#           }

import re
import typing as tp
"""
@Callback: Counter可以看作是将字符映射到出现次数上的dictionary; 
           @Func{update}将更新参数中所有字符的出现次数，可以直接在元素上做简单的加减法
           @Func{most_common}(n), 根据参数返回前n个频率最高的符号及其频率够成的List, 默认从大到小返回所有的符号
           支持两个Counter之间的加减法, 类似于集合的交并。
"""
from collections import Counter
"""
@Callback:  deque是栈与队列的广义实现, 支持线程安全, 内存高效, 以近似O(1)的性能在deque的两端插入/删除元素
            List在pop(0), insert(0, v)这种会改变数据位置与大小的操作上会有O(n)的复杂度
            @Func{append}, @Func{appendleft}, @Func{extend}, @Func{extendleft}, @Func{pop}, @Func{popleft}
            @Func{count}, @Func{insert}(insert, obj), @Func{rotate}, @Func{clear}, @Func{remove}, @Func{maxlen}
            元素可以异构
"""
from collections import deque
from dataclasses import dataclass
#@Question: How to use @Lib{bitarray}?
from bitarray import bitarray, util
from fairseq.data import Dictionary



#@Desc: basically we have to write to addressable bytes for the memory mapped dataset loader.
#       Sentences that get encoded to a length that is not a multiple of BLOCKSIZE (a byte) will be padded to fit. (see _pad in the coder)
BLOCKSIZE = 8



class HuffmanCoder:
    def __init__(
        self, root: "HuffmanNode", bos="<s>", pad="<pad>", eos="</s>", unk="<unk>"
    ):
        self.root = root
        self.table = root.code_table()
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos

    #@Desc: bitpadding after the code, 1 then some 0s. If the array is already a multiple of blocksize, we add a full block.
    def _pad(self, a: bitarray) -> bitarray:
        pad_len = BLOCKSIZE - (len(a) % BLOCKSIZE) - 1
        padding = bitarray("1" + "0" * pad_len)
        return a + padding

    #@Desc: remove the bitpadding after the code. There will be a set of 0s preceded by a 1 at the end of the bitarray, we remove that
    def _unpad(self, a: bitarray) -> bitarray:
        #@Explain: count the 0 padding at the end until we find the first 1, we want to remove the one too
        remove_cnt = util.rindex(a, 1)
        return a[:remove_cnt]

    #@Desc: encode a list of tokens a return bytes. We use bitpadding to make sure the encoded bits fit in bytes.
    #       将一组tokens的code拼在一起
    def encode(self, iter: tp.List[str]) -> bytes:
        a = bitarray()
        for token in iter:
            code = self.get_code(token)
            if code is None:
                if self.unk_word is None:
                    raise Exception(f"unknown token {token} cannot be encoded.")
                else:
                    token = self.unk_word
            #@Explain: 由于token可能是unknown word，因此再次获取其code
            a = a + self.get_code(token)
        return self._pad(a).tobytes()

    #@Desc: take bitpadded bytes and decode it to a set of leaves. You can then use each node to find the symbol/id
    #       由于Huffman Tree的叶子节点是具体的信息，因此给定一组字节以及一个Huffman Tree，能够循环地解码出所有的信息！
    #@Return: @Type{Iterator["HuffmanNode"]}
    def decode(self, bits: bytes) -> tp.Iterator["HuffmanNode"]:
        a = bitarray()
        a.frombytes(bits)
        return self.root.decode(self._unpad(a))

    #@Desc: Call @Func{get_noce} and then return @Type{HuffmanNode}.code
    def get_code(self, symbol: str) -> tp.Optional[bitarray]:
        node = self.get_node(symbol)
        return None if node is None else node.code

    #@Desc: return @Type{HuffmanNode} based on @Param{symbol} according to @Var{self.table}
    def get_node(self, symbol: str) -> "HuffmanNode":
        return self.table.get(symbol)

    #@Desc: Factory method from file.
    #@Callback: the static method in Python Class has a @Param{cls} which represent the Class.
    #@Return: An instance of @Type{HuffmanCoder}
    @classmethod
    def from_file(
        cls,
        filename: str,
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
    ) -> "HuffmanCoder":
        builder = HuffmanCodeBuilder.from_file(filename)
        return builder.build_code(bos=bos, pad=pad, eos=eos, unk=unk)

    #@Desc: Write the code table to a file in a specific format.
    def to_file(self, filename, sep="\t"):
        nodes = list(self.table.values())
        #@Now: @Var{nodes} is a list of @Type{Dictionary{str: HuffmanNode}}
        #@Explain: 在构建Huffman Tree时，叶子节点的id越小，频率越高, @Depend{@Class{HuffmanBuilder}.build_code}
        nodes.sort(key=lambda n: n.id)
        with open(filename, "w", encoding="utf-8") as output:
            for n in nodes:
                output.write(f"{n.symbol}{sep}{n.count}\n")
            #@Now: 文件格式与@Class{HuffmanBuilder}.to_file是一致的， @Depend{@Class{HuffmanBuilder}.to_file}

    #@Desc: 遍历即为遍历@Var{self.code_table}, which is @Type{Dictionary{str: HuffmanNode}}
    def __iter__(self):
        for n in self.table.values():
            yield n

    #@Desc: Merge with the other coder.
    #@Return: Reconstruct a HuffmanCoder!
    def merge(self, other_coder: "HuffmanCoder") -> "HuffmanCoder":
        builder = HuffmanCodeBuilder()
        for n in self:
            builder.increment(n.symbol, n.count)
        for n in other_coder:
            builder.increment(n.symbol, n.count)
        return builder.build_code()

    #@Desc: 判断相等性时只需要判断code table是否一致，不会看Huffman Tree
    def __eq__(self, other: "HuffmanCoder") -> bool:
        return self.table == other.table

    def __len__(self) -> int:
        return len(self.table)

    def __contains__(self, sym: str) -> bool:
        return sym in self.table

    def to_dictionary(self) -> Dictionary:
        dictionary = Dictionary(bos=self.bos, unk=self.unk, pad=self.pad, eos=self.eos)
        for n in self:
            dictionary.add_symbol(n.symbol, n=n.count)
        dictionary.finalize()
        return dictionary



"""
@Desc: a node in a Huffman tree
@Note: @Param{symbol} 不一定是一个字符, 它可以是任何str
"""
@dataclass
class HuffmanNode:
    id: int
    count: int
    symbol: tp.Optional[str] = None
    left: tp.Optional["HuffmanNode"] = None
    right: tp.Optional["HuffmanNode"] = None
    code: tp.Optional[bitarray] = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    #@Desc: Encode the leaf nodes on the Huffman tree according to the Huffman coding rules.
    #@Return: @Type{Dictionary{str: HuffmanNode}}, result of Huffman coding.
    def code_table(
        self, prefix: tp.Optional[bitarray] = None
    ) -> tp.Dict[str, "HuffmanNode"]:
        defaulted_prefix = prefix if prefix is not None else bitarray()
        if self.is_leaf():
            self.code = (
                #@Explain: leaf could be the root if there is only one symbol
                defaulted_prefix if len(defaulted_prefix) > 0 else bitarray("0")
            )  
            return {self.symbol: self}

        codes_right = self.right.code_table(defaulted_prefix + bitarray([0]))
        codes_left = self.left.code_table(defaulted_prefix + bitarray([1]))
        #@Question: How to use double star symbol?
        return {**codes_left, **codes_right}

    def decode(self, bits: bitarray) -> tp.Iterator["HuffmanNode"]:
        current_node = self
        for bit in bits:
            #@Explain: 0 go right, 1 go left
            if bit == 0:
                current_node = current_node.right
            else:
                current_node = current_node.left
            #@Explain: we shouldn't be on a leaf here.
            if current_node is None:
                raise Exception("fell off a leaf")

            if current_node.is_leaf():
                #@Question: How to use @Keyward{yield}?
                yield current_node
                current_node = self
        #@Now: At the end, the current node should be a leaf.
        if current_node != self:
            raise Exception("couldn't decode all the bits")



"""
@Desc: build a dictionary with occurence count and then build the Huffman code for it.
"""
class HuffmanCodeBuilder:
    def __init__(self):
        self.symbols = Counter()

    def add_symbols(self, *syms) -> None:
        self.symbols.update(syms)

    def increment(self, symbol: str, cnt: int) -> None:
        self.symbols[symbol] += cnt

    #@Desc: Factory method from file.
    #@Callback: the static method in Python Class has a @Param{cls} which represent the Class.
    @classmethod
    def from_file(cls, filename):
        #@Explain: Initialize an instance of @Class{HuffmanCodeBuilder}
        c = cls()
        with open(filename, "r", encoding="utf-8") as input:
            for line in input:
                split = re.split(r"[\s]+", line)
                c.increment(split[0], int(split[1]))
        return c

    #@Desc: Write the dictionary to a file in a specific format.
    def to_file(self, filename, sep="\t"):
        with open(filename, "w", encoding="utf-8") as output:
            for (tok, cnt) in self.symbols.most_common():
                output.write(f"{tok}{sep}{cnt}\n")
    
    #@Param{q1, q2}: The Element must be @Type{HuffmanNode} and they should be sorted.
    #@Return: the smallest element of @Param{q1} and @Param{q2}
    def _smallest(self, q1: deque, q2: deque) -> HuffmanNode:
        if len(q1) == 0:
            return q2.pop()

        if len(q2) == 0:
            return q1.pop()

        if q1[-1].count < q2[-1].count:
            return q1.pop()

        return q2.pop()

    #@Desc: combine two Counter, and REconstruct an instance of @Class{HuffmanCodeBuilder}
    def __add__(self, c: "HuffmanCodeBuilder") -> "HuffmanCodeBuilder":
        new_c = self.symbols + c.symbols
        new_b = HuffmanCodeBuilder()
        new_b.symbols = new_c
        return new_b

    #@Desc: Construct a Huffman Tree according to @Var{self.symbols}
    #@Return: @Type{HuffmanCoder} which @Var{root} is the huffman tree.
    def build_code(
        self,
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
    ) -> HuffmanCoder:
        assert len(self.symbols) > 0, "cannot build code from empty list of symbols"

        #@Explain: Make sure specific symbols are existed.
        if self.symbols[bos] == 0:
            self.add_symbols(bos)
        if self.symbols[pad] == 0:
            self.add_symbols(pad)
        if self.symbols[eos] == 0:
            self.add_symbols(eos)
        if self.symbols[unk] == 0:
            self.add_symbols(unk)

        #@Explain:  Initialize HuffmanNode in a deque based on their frequency. 
        #           left are the most common, right are the least common
        node_id = 0
        leaves_queue = deque(
            [
                HuffmanNode(symbol=symbol, count=count, id=idx)
                for idx, (symbol, count) in enumerate(self.symbols.most_common())
            ]
        )
        
        #@Explain: Special case: only one symbol! In fact, The situation is rare.
        if len(leaves_queue) == 1:
            root = leaves_queue.pop()
            root.id = 0
            return HuffmanCoder(root)

        nodes_queue = deque()

        #@Desc: Construct a Huffman tree according to Huffman coding rules.
        while len(leaves_queue) > 0 or len(nodes_queue) != 1:
            #@Expalin: get the lowest two nodes at the head of each queue
            node1 = self._smallest(leaves_queue, nodes_queue)
            node2 = self._smallest(leaves_queue, nodes_queue)

            #@Explain: add new node to @Var{nodes_queue}. It's right to insert the left of queue directly.
            nodes_queue.appendleft(
                HuffmanNode(
                    count=node1.count + node2.count, left=node1, right=node2, id=node_id
                )
            )
            node_id += 1

        #@Now: We have a Huffman Tree.
        #@Explain: construct the HuffmanCoder, we are left with the root.
        return HuffmanCoder(nodes_queue.pop(), bos=bos, pad=pad, eos=eos, unk=unk)