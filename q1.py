import random

s_box = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

inv_s_box = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)


def sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = s_box[s[i][j]]


def inv_sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = inv_s_box[s[i][j]]


def shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]


def inv_shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]

def add_round_key(s, k):
    for i in range(4):
        for j in range(4):
            s[i][j] ^= k[i][j]


# learned from https://web.archive.org/web/20100626212235/http://cs.ucsb.edu/~koc/cs178/projects/JT/aes.c
xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)


def mix_single_column(a):
    # see Sec 4.1.2 in The Design of Rijndael
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)


def mix_columns(s):
    for i in range(4):
        mix_single_column(s[i])


def inv_mix_columns(s):
    # see Sec 4.1.3 in The Design of Rijndael
    for i in range(4):
        u = xtime(xtime(s[i][0] ^ s[i][2]))
        v = xtime(xtime(s[i][1] ^ s[i][3]))
        s[i][0] ^= u
        s[i][1] ^= v
        s[i][2] ^= u
        s[i][3] ^= v

    mix_columns(s)


r_con = (
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
)


def bytes2matrix(text):
    """ Converts a 16-byte array into a 4x4 matrix.  """
    return [list(text[i:i+4]) for i in range(0, len(text), 4)]

def matrix2bytes(matrix):
    """ Converts a 4x4 matrix into a 16-byte array.  """
    return bytes(sum(matrix, []))

def xor_bytes(a, b):
    """ Returns a new byte array with the elements xor'ed. """
    return bytes(i^j for i, j in zip(a, b))

def inc_bytes(a):
    """ Returns a new byte array with the value increment by 1 """
    out = list(a)
    for i in reversed(range(len(out))):
        if out[i] == 0xFF:
            out[i] = 0
        else:
            out[i] += 1
            break
    return bytes(out)

def pad(plaintext):
    """
    Pads the given plaintext with PKCS#7 padding to a multiple of 16 bytes.
    Note that if the plaintext size is a multiple of 16,
    a whole block will be added.
    """
    padding_len = 16 - (len(plaintext) % 16)
    padding = bytes([padding_len] * padding_len)
    return plaintext + padding

def unpad(plaintext):
    """
    Removes a PKCS#7 padding, returning the unpadded text and ensuring the
    padding was correct.
    """
    padding_len = plaintext[-1]
    assert padding_len > 0
    message, padding = plaintext[:-padding_len], plaintext[-padding_len:]
    assert all(p == padding_len for p in padding)
    return message

def split_blocks(message, block_size=16, require_padding=True):
        assert len(message) % block_size == 0 or not require_padding
        return [message[i:i+16] for i in range(0, len(message), block_size)]

class AES:
    """
    Class for AES-128 encryption with CBC mode and PKCS#7.

    This is a raw implementation of AES, without key stretching or IV
    management. Unless you need that, please use `encrypt` and `decrypt`.
    """
    rounds_by_key_size = {16: 10, 24: 12, 32: 14}
    def __init__(self, master_key):
        """
        Initializes the object with a given key.
        """
        assert len(master_key) in AES.rounds_by_key_size
        self.n_rounds = AES.rounds_by_key_size[len(master_key)]
        self._key_matrices = self._expand_key(master_key)

    def _expand_key(self, master_key):
        """
        Expands and returns a list of key matrices for the given master_key.
        """
        # Initialize round keys with raw key material.
        key_columns = bytes2matrix(master_key)
        iteration_size = len(master_key) // 4

        i = 1
        while len(key_columns) < (self.n_rounds + 1) * 4:
            # Copy previous word.
            word = list(key_columns[-1])

            # Perform schedule_core once every "row".
            if len(key_columns) % iteration_size == 0:
                # Circular shift.
                word.append(word.pop(0))
                # Map to S-BOX.
                word = [s_box[b] for b in word]
                # XOR with first byte of R-CON, since the others bytes of R-CON are 0.
                word[0] ^= r_con[i]
                i += 1
            elif len(master_key) == 32 and len(key_columns) % iteration_size == 4:
                # Run word through S-box in the fourth iteration when using a
                # 256-bit key.
                word = [s_box[b] for b in word]

            # XOR with equivalent word from previous iteration.
            word = xor_bytes(word, key_columns[-iteration_size])
            key_columns.append(word)

        # Group key words in 4x4 byte matrices.
        return [key_columns[4*i : 4*(i+1)] for i in range(len(key_columns) // 4)]

    def encrypt_block(self, plaintext):
        """
        Encrypts a single block of 16 byte long plaintext.
        """
        assert len(plaintext) == 16

        plain_state = bytes2matrix(plaintext)

        add_round_key(plain_state, self._key_matrices[0])

        for i in range(1, self.n_rounds):
            sub_bytes(plain_state)
            shift_rows(plain_state)
            mix_columns(plain_state)
            add_round_key(plain_state, self._key_matrices[i])

        sub_bytes(plain_state)
        shift_rows(plain_state)
        add_round_key(plain_state, self._key_matrices[-1])

        return matrix2bytes(plain_state)

    def decrypt_block(self, ciphertext):
        """
        Decrypts a single block of 16 byte long ciphertext.
        """
        assert len(ciphertext) == 16

        cipher_state = bytes2matrix(ciphertext)

        add_round_key(cipher_state, self._key_matrices[-1])
        inv_shift_rows(cipher_state)
        inv_sub_bytes(cipher_state)

        for i in range(self.n_rounds - 1, 0, -1):
            add_round_key(cipher_state, self._key_matrices[i])
            inv_mix_columns(cipher_state)
            inv_shift_rows(cipher_state)
            inv_sub_bytes(cipher_state)

        add_round_key(cipher_state, self._key_matrices[0])

        return matrix2bytes(cipher_state)

    def encrypt_ecb(self, plaintext):
        """
        Encrypts `plaintext` using CBC mode and PKCS#7 padding, with the given
        initialization vector (iv).
        """
        plaintext = pad(plaintext)

        blocks = []
        for plaintext_block in split_blocks(plaintext):
            # CBC mode encrypt: encrypt(plaintext_block XOR previous)
            block = self.encrypt_block(plaintext_block)
            blocks.append(block)

        return b''.join(blocks)

    def decrypt_ecb(self, ciphertext):
        """
        Decrypts `ciphertext` using ECB mode and PKCS#7 padding, with the given
        initialization vector (iv).
        """

        blocks = []
        for ciphertext_block in split_blocks(ciphertext):
            # ECB mode decrypt: previous XOR decrypt(ciphertext)
            blocks.append(self.decrypt_block(ciphertext_block))

        return unpad(b''.join(blocks))

    def encrypt_cbc(self, plaintext, iv):
        """
        Encrypts `plaintext` using CBC mode and PKCS#7 padding, with the given
        initialization vector (iv).
        """
        assert len(iv) == 16

        plaintext = pad(plaintext)

        blocks = []
        previous = iv
        for plaintext_block in split_blocks(plaintext):
            # CBC mode encrypt: encrypt(plaintext_block XOR previous)
            block = self.encrypt_block(xor_bytes(plaintext_block, previous))
            blocks.append(block)
            previous = block

        return b''.join(blocks)

    def decrypt_cbc(self, ciphertext, iv):
        """
        Decrypts `ciphertext` using CBC mode and PKCS#7 padding, with the given
        initialization vector (iv).
        """
        assert len(iv) == 16

        blocks = []
        previous = iv
        for ciphertext_block in split_blocks(ciphertext):
            # CBC mode decrypt: previous XOR decrypt(ciphertext)
            blocks.append(xor_bytes(previous, self.decrypt_block(ciphertext_block)))
            previous = ciphertext_block

        return unpad(b''.join(blocks))

    def encrypt_cfb(self, plaintext, iv):
        """
        Encrypts `plaintext` with the given initialization vector (iv).
        """
        assert len(iv) == 16

        blocks = []
        prev_ciphertext = iv
        for plaintext_block in split_blocks(plaintext, require_padding=False):
            # CFB mode encrypt: plaintext_block XOR encrypt(prev_ciphertext)
            ciphertext_block = xor_bytes(plaintext_block, self.encrypt_block(prev_ciphertext))
            blocks.append(ciphertext_block)
            prev_ciphertext = ciphertext_block

        return b''.join(blocks)

    def decrypt_cfb(self, ciphertext, iv):
        """
        Decrypts `ciphertext` with the given initialization vector (iv).
        """
        assert len(iv) == 16

        blocks = []
        prev_ciphertext = iv
        for ciphertext_block in split_blocks(ciphertext, require_padding=False):
            # CFB mode decrypt: ciphertext XOR decrypt(prev_ciphertext)
            plaintext_block = xor_bytes(ciphertext_block, self.encrypt_block(prev_ciphertext))
            blocks.append(plaintext_block)
            prev_ciphertext = ciphertext_block

        return b''.join(blocks)

    def encrypt_ofb(self, plaintext, iv):
        """
        Encrypts `plaintext` using OFB mode initialization vector (iv).
        """
        assert len(iv) == 16

        blocks = []
        previous = iv
        for plaintext_block in split_blocks(plaintext, require_padding=False):
            # OFB mode encrypt: plaintext_block XOR encrypt(previous)
            block = self.encrypt_block(previous)
            ciphertext_block = xor_bytes(plaintext_block, block)
            blocks.append(ciphertext_block)
            previous = block

        return b''.join(blocks)

    def decrypt_ofb(self, ciphertext, iv):
        """
        Decrypts `ciphertext` using OFB mode initialization vector (iv).
        """
        assert len(iv) == 16

        blocks = []
        previous = iv
        for ciphertext_block in split_blocks(ciphertext, require_padding=False):
            # OFB mode decrypt: ciphertext XOR encrypt(previous)
            block = self.encrypt_block(previous)
            plaintext_block = xor_bytes(ciphertext_block, block)
            blocks.append(plaintext_block)
            previous = block

        return b''.join(blocks)

    def encrypt_ctr(self, plaintext, iv):
        """
        Encrypts `plaintext` using CTR mode with the given nounce/IV.
        """
        assert len(iv) == 16

        blocks = []
        nonce = iv
        for plaintext_block in split_blocks(plaintext, require_padding=False):
            # CTR mode encrypt: plaintext_block XOR encrypt(nonce)
            block = xor_bytes(plaintext_block, self.encrypt_block(nonce))
            blocks.append(block)
            nonce = inc_bytes(nonce)

        return b''.join(blocks)

    def decrypt_ctr(self, ciphertext, iv):
        """
        Decrypts `ciphertext` using CTR mode with the given nounce/IV.
        """
        assert len(iv) == 16

        blocks = []
        nonce = iv
        for ciphertext_block in split_blocks(ciphertext, require_padding=False):
            # CTR mode decrypt: ciphertext XOR encrypt(nonce)
            block = xor_bytes(ciphertext_block, self.encrypt_block(nonce))
            blocks.append(block)
            nonce = inc_bytes(nonce)

        return b''.join(blocks)

class DES:

  # Table of Position of 64 bits at initial level: Initial Permutation Table
  initial_perm = [58, 50, 42, 34, 26, 18, 10, 2,
          60, 52, 44, 36, 28, 20, 12, 4,
          62, 54, 46, 38, 30, 22, 14, 6,
          64, 56, 48, 40, 32, 24, 16, 8,
          57, 49, 41, 33, 25, 17, 9, 1,
          59, 51, 43, 35, 27, 19, 11, 3,
          61, 53, 45, 37, 29, 21, 13, 5,
          63, 55, 47, 39, 31, 23, 15, 7]

  # Expansion D-box Table
  exp_d = [32, 1, 2, 3, 4, 5, 4, 5,
      6, 7, 8, 9, 8, 9, 10, 11,
      12, 13, 12, 13, 14, 15, 16, 17,
      16, 17, 18, 19, 20, 21, 20, 21,
      22, 23, 24, 25, 24, 25, 26, 27,
      28, 29, 28, 29, 30, 31, 32, 1]

  # Straight Permutation Table
  per = [16, 7, 20, 21,
    29, 12, 28, 17,
    1, 15, 23, 26,
    5, 18, 31, 10,
    2, 8, 24, 14,
    32, 27, 3, 9,
    19, 13, 30, 6,
    22, 11, 4, 25]

  # S-box Table
  sbox = [[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
      [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
      [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
      [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],

      [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
      [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
      [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
      [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],

      [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
      [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
      [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
      [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],

      [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
      [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
      [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
      [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],

      [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
      [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
      [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
      [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],

      [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
      [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
      [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
      [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],

      [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
      [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
      [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
      [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],

      [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
      [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
      [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
      [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]]

  # Final Permutation Table
  final_perm = [40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25]
  
  def hex2bin(self, s):
    mp = {'0': "0000",
      '1': "0001",
      '2': "0010",
      '3': "0011",
      '4': "0100",
      '5': "0101",
      '6': "0110",
      '7': "0111",
      '8': "1000",
      '9': "1001",
      'A': "1010",
      'B': "1011",
      'C': "1100",
      'D': "1101",
      'E': "1110",
      'F': "1111"}
    bin = ""
    for i in range(len(s)):
      bin = bin + mp[s[i]]
    return bin

  # Binary to hexadecimal conversion

  def bin2hex(self, s):
    mp = {"0000": '0',
      "0001": '1',
      "0010": '2',
      "0011": '3',
      "0100": '4',
      "0101": '5',
      "0110": '6',
      "0111": '7',
      "1000": '8',
      "1001": '9',
      "1010": 'A',
      "1011": 'B',
      "1100": 'C',
      "1101": 'D',
      "1110": 'E',
      "1111": 'F'}
    hex = ""
    for i in range(0, len(s), 4):
      ch = ""
      ch = ch + s[i]
      ch = ch + s[i + 1]
      ch = ch + s[i + 2]
      ch = ch + s[i + 3]
      hex = hex + mp[ch]

    return hex
  
  def hex_conv(self, text):
    hex_val = ''
    for i in range(0,8):
      val = ord(text[i])
      rem = val//16 
      rem = rem + 55 if rem > 9 else rem + 48
      hex_val += chr(rem)
      rem = val%16 
      rem = rem + 55 if rem > 9 else rem + 48
      hex_val += chr(rem)
    return hex_val

  def rev_hex_conv(self, text):
    hex_val = ''
    for i in range(0,16,2):
      val = ord(text[i])
      val = val - 65 + 10 if val > 57 else val - 48
      val *= 16
      val1 = ord(text[i+1])
      val1 = val1 - 65 + 10 if val1 > 57 else val1 - 48
      hex_val += chr(val + val1)
    return hex_val
      
  def bin2dec(self, binary):

    binary1 = binary
    decimal, i, n = 0, 0, 0
    while(binary != 0):
      dec = binary % 10
      decimal = decimal + dec * pow(2, i)
      binary = binary//10
      i += 1
    return decimal

  # Decimal to binary conversion

  def dec2bin(self, num):
    res = bin(num).replace("0b", "")
    if(len(res) % 4 != 0):
      div = len(res) / 4
      div = int(div)
      counter = (4 * (div + 1)) - len(res)
      for i in range(0, counter):
        res = '0' + res
    return res
  # Permute function to rearrange the bits

  def permute(self, k, arr, n):
    permutation = ""
    for i in range(0, n):
      permutation = permutation + k[arr[i] - 1]
    return permutation

  # shifting the bits towards left by nth shifts

  def shift_left(self, k, nth_shifts):
    s = ""
    for i in range(nth_shifts):
      for j in range(1, len(k)):
        s = s + k[j]
      s = s + k[0]
      k = s
      s = ""
    return k

  # calculating xor of two strings of binary number a and b

  def xor(self, a, b):
    ans = ""
    for i in range(len(a)):
      if a[i] == b[i]:
        ans = ans + "0"
      else:
        ans = ans + "1"
    return ans

  def __init__(self,deskey=None):
    if deskey == None:
      rand_key = "ABCDEFGHIJKLMOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"
      deskey = ''
      for i in range(0,8):
        deskey += rand_key[random.randrange(60)]
    self.rkb = []
    self.rk = []
    self.drkb = []
    self.drk = []
    self.act_key = ''
    self.act_key = deskey
    hex_key = self.hex_conv(deskey)
    key = self.hex2bin(hex_key)

    # --parity bit drop table
    keyp = [57, 49, 41, 33, 25, 17, 9,
        1, 58, 50, 42, 34, 26, 18,
        10, 2, 59, 51, 43, 35, 27,
        19, 11, 3, 60, 52, 44, 36,
        63, 55, 47, 39, 31, 23, 15,
        7, 62, 54, 46, 38, 30, 22,
        14, 6, 61, 53, 45, 37, 29,
        21, 13, 5, 28, 20, 12, 4]

    # getting 56 bit key from 64 bit using the parity bits
    key = self.permute(key, keyp, 56)

    # Number of bit shifts
    shift_table = [1, 1, 2, 2,
          2, 2, 2, 2,
          1, 2, 2, 2,
          2, 2, 2, 1]

    # Key- Compression Table : Compression of key from 56 bits to 48 bits
    key_comp = [14, 17, 11, 24, 1, 5,
          3, 28, 15, 6, 21, 10,
          23, 19, 12, 4, 26, 8,
          16, 7, 27, 20, 13, 2,
          41, 52, 31, 37, 47, 55,
          30, 40, 51, 45, 33, 48,
          44, 49, 39, 56, 34, 53,
          46, 42, 50, 36, 29, 32]

    # Splitting
    left = key[0:28] # rkb for RoundKeys in binary
    right = key[28:56] # rk for RoundKeys in hexadecimal

    for i in range(0, 16):
      # Shifting the bits by nth shifts by checking from shift table
      left = self.shift_left(left, shift_table[i])
      right = self.shift_left(right, shift_table[i])

      # Combination of left and right string
      combine_str = left + right

      # Compression of key from 56 to 48 bits
      round_key = self.permute(combine_str, key_comp, 48)
      # print("round-",i," key: ",round_key)
      self.rkb.append(round_key)
      self.rk.append(self.bin2hex(round_key))
    self.drkb = list(reversed(self.rkb))
    self.drk = list(reversed(self.rk))
  
  def pad(self,plainText):
    add_x = 8 - len(plainText) % 8
    if add_x == 0: 
      add_x = 8
    for i in range(0,add_x-1):
      plainText += '_'
    plainText += chr(48 + add_x)
    return plainText
  
  def unpad(self,plainText):
    pad = ord(plainText[-1]) - 48
    plainText = plainText[0:-pad]
    return plainText

  def encrypt_ECB(self,plainText):
    cypherText = '0x'

    # padding 
    plainText = self.pad(plainText)

    # divide into blocks
    for i in range(0,len(plainText),8):
      pt = plainText[i:i+8]
      pt = self.hex_conv(pt)
      pt = self.encrypt(pt,0)
      cypherText += pt
    return cypherText, self.act_key

  def decrypt_ECB(self,cypherText, u_pad=1):
    plainText = ''
    cypherText = cypherText[2:len(cypherText)] 

    # divide into blocks
    for i in range(0,len(cypherText),16):
      ct = cypherText[i:i+16]
      ct = self.encrypt(ct,1)
      ct = self.rev_hex_conv(ct)
      plainText += ct

    # unpadding 
    if u_pad == 1:
      plainText = self.unpad(plainText)   
    
    return plainText
  
  def encrypt_cbc(self, plaintext, iv):
        """
        Encrypts `plaintext` using CBC mode and PKCS#7 padding, with the given
        initialization vector (iv).
        """
        assert len(iv) == 8

        plaintext = self.pad(plaintext)

        cypherText = '0x'
        previous = self.hex_conv(iv)
        for i in range(0,len(plaintext),8):
            # CBC mode encrypt: encrypt(plaintext_block XOR previous)
            pt = plaintext[i:i+8]
            pt = self.hex_conv(pt)
            pt = self.bin2hex(self.xor(self.hex2bin(pt), self.hex2bin(previous)))
            block = self.encrypt(pt,0)
            cypherText += block
            previous = block

        return cypherText, self.act_key

  def decrypt_cbc(self, cyphertext, iv, u_pad=1):
      """
      Decrypts `ciphertext` using CBC mode and PKCS#7 padding, with the given
      initialization vector (iv).
      """
      assert len(iv) == 8
      cyphertext = cyphertext[2:len(cyphertext)] 
      plaintext = ''
      previous = self.hex_conv(iv)
      for i in range(0,len(cyphertext),16):
          # CBC mode decrypt: previous XOR decrypt(ciphertext)
          ct = cyphertext[i:i+16]
          dct = self.encrypt(ct,1)
          pt = self.xor(self.hex2bin(previous), self.hex2bin(dct))
          pt = self.bin2hex(pt)
          pt = self.rev_hex_conv(pt)
          plaintext += pt
          previous = ct

      # unpadding 
      if u_pad == 1:
        plaintext = self.unpad(plaintext)
      return plaintext

  def encrypt_cfb(self, plaintext, iv):
      """
      Encrypts `plaintext` with the given initialization vector (iv).
      """
      assert len(iv) == 8
      plaintext = self.pad(plaintext)
      cyphertext = '0x'
      previous = self.hex_conv(iv)
      for i in range(0,len(plaintext),8):
          # CFB mode encrypt: plaintext_block XOR encrypt(prev_ciphertext)
          pt = plaintext[i:i+8]
          pt = self.hex2bin(self.hex_conv(pt)) 
          dpt = self.encrypt(previous,0)
          ciphertext_block = self.xor(pt, self.hex2bin(dpt))
          ciphertext_block = self.bin2hex(ciphertext_block)
          cyphertext += ciphertext_block
          previous = ciphertext_block

      return cyphertext, self.act_key

  def decrypt_cfb(self, cyphertext, iv, u_pad=1):
      """
      Decrypts `ciphertext` with the given initialization vector (iv).
      """
      assert len(iv) == 8
      cyphertext = cyphertext[2:len(cyphertext)]
      plaintext = ''
      previous = self.hex_conv(iv)
      for i in range(0,len(cyphertext),16):
          # CFB mode decrypt: ciphertext XOR decrypt(prev_ciphertext)
          ct = cyphertext[i:i+16]
          bct = self.encrypt(previous,0)
          pt= self.xor(self.hex2bin(ct), self.hex2bin(bct))
          pt = self.bin2hex(pt)
          pt = self.rev_hex_conv(pt)
          plaintext += pt
          previous = ct

      if u_pad == 1:
        plaintext = self.unpad(plaintext)
      return plaintext

  def encrypt_ofb(self, plaintext, iv):
      """
      Encrypts `plaintext` using OFB mode initialization vector (iv).
      """
      assert len(iv) == 8
      plaintext = self.pad(plaintext)
      cyphertext = '0x'
      previous = self.hex_conv(iv)
      for i in range(0,len(plaintext),8):
          # OFB mode encrypt: plaintext_block XOR encrypt(previous)
          block = self.encrypt(previous,0)
          pt = plaintext[i:i+8]
          pt = self.hex2bin(self.hex_conv(pt))
          ciphertext_block = self.xor(pt, self.hex2bin(block))
          ciphertext_block = self.bin2hex(ciphertext_block)
          cyphertext += ciphertext_block
          previous = block
      return cyphertext, self.act_key

  def decrypt_ofb(self, cyphertext, iv, u_pad=1):
      """
      Decrypts `ciphertext` using OFB mode initialization vector (iv).
      """
      assert len(iv) == 8
      cyphertext = cyphertext[2:len(cyphertext)]
      plaintext = ''
      previous = self.hex_conv(iv)
      for i in range(0,len(cyphertext),16):
          # OFB mode decrypt: ciphertext XOR encrypt(previous)
          ct = cyphertext[i:i+16]
          bct = self.encrypt(previous,0)
          pt= self.xor(self.hex2bin(ct), self.hex2bin(bct))
          pt = self.bin2hex(pt)
          pt = self.rev_hex_conv(pt)
          plaintext += pt
          previous = bct
      if u_pad == 1:
        plaintext = self.unpad(plaintext)
      return plaintext

  def inc_bytes(self,a):
      hexs = '0123456789ABCDEF'
      out = ''
      for i in range(0,16,2):
         if a[i:i+2] == 'FF':
             out += '00'
         else:    
             val = ord(a[i+1])
             val = val - 65 + 10 if val > 57 else val - 48
             if val<15:
                out += a[i]+hexs[val+1]
             else:
                val = ord(a[i])
                val = val - 65 + 10 if val > 57 else val - 48
                out += hexs[val+1] + '0'
      return out


  def encrypt_ctr(self, plaintext, iv):
      """
      Encrypts `plaintext` using CTR mode with the given nounce/IV.
      """
      assert len(iv) == 8
      plaintext = self.pad(plaintext)
      cyphertext = '0x'
      nonce = self.hex_conv(iv)
      for i in range(0,len(plaintext),8):
          # CTR mode encrypt: plaintext_block XOR encrypt(nonce)
          pt = plaintext[i:i+8]
          pt = self.hex2bin(self.hex_conv(pt)) 
          dpt = self.encrypt(nonce,0)
          ciphertext_block = self.xor(pt, self.hex2bin(dpt))
          ciphertext_block = self.bin2hex(ciphertext_block)
          cyphertext += ciphertext_block
          nonce = self.inc_bytes(nonce)
      return cyphertext, self.act_key


  def decrypt_ctr(self, cyphertext, iv, u_pad=1):
      """
      Decrypts `ciphertext` using CTR mode with the given nounce/IV.
      """
      assert len(iv) == 8
      cyphertext = cyphertext[2:len(cyphertext)]
      plaintext = ''
      nonce = self.hex_conv(iv)
      for i in range(0,len(cyphertext),16):
          # CTR mode decrypt: ciphertext XOR encrypt(nonce)
          ct = cyphertext[i:i+16]
          bct = self.encrypt(nonce,0)
          pt= self.xor(self.hex2bin(ct), self.hex2bin(bct))
          pt = self.bin2hex(pt)
          pt = self.rev_hex_conv(pt)
          plaintext += pt
          nonce = self.inc_bytes(nonce)
      if u_pad == 1:
        plaintext = self.unpad(plaintext)
      return plaintext


  def encrypt(self,pt,decrypt):
    pt = self.hex2bin(pt)

    # Initial Permutation
    pt = self.permute(pt, self.initial_perm, 64)
    print("After initial permutation", self.bin2hex(pt))

    if decrypt==0:
      rkb = self.rkb
      rk = self.rk
    else:
      rkb = self.drkb
      rk = self.drk

    # Splitting
    left = pt[0:32]
    right = pt[32:64]
    for i in range(0, 16):
      # Expansion D-box: Expanding the 32 bits data into 48 bits
      right_expanded = self.permute(right, self.exp_d, 48)

      # XOR RoundKey[i] and right_expanded
      xor_x = self.xor(right_expanded, rkb[i])

      # S-boxex: substituting the value from s-box table by calculating row and column
      sbox_str = ""
      for j in range(0, 8):
        row = self.bin2dec(int(xor_x[j * 6] + xor_x[j * 6 + 5]))
        col = self.bin2dec(int(xor_x[j * 6 + 1] + xor_x[j * 6 + 2] + xor_x[j * 6 + 3] + xor_x[j * 6 + 4]))
        val = self.sbox[j][row][col]
        sbox_str = sbox_str + self.dec2bin(val)

      # Straight D-box: After substituting rearranging the bits
      sbox_str = self.permute(sbox_str, self.per, 32)

      # XOR left and sbox_str
      result = self.xor(left, sbox_str)
      left = result

      # Swapper
      if(i != 15):
        left, right = right, left
      print("Round ", i + 1, " ", self.bin2hex(left),
        " ", self.bin2hex(right), " ", rk[i])

    # Combination
    combine = left + right

    # Final permutation: final rearranging of bits to get cipher text
    cipher_text = self.permute(combine, self.final_perm, 64)
    cipher_text = self.bin2hex(cipher_text)
    return cipher_text

class TDES:

  def __init__(self,deskey):
    if deskey == None:
      rand_key = "ABCDEFGHIJKLMOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"
      deskey = ''
      for i in range(0,24):
        deskey += rand_key[random.randrange(60)]
    self.act_key = []
    self.act_key.append(deskey[0:8])
    self.act_key.append(deskey[8:16])
    self.act_key.append(deskey[16:24])
  
  def hex_conv(self, text):
    hex_val = ''
    for i in range(0,8):
      val = ord(text[i])
      rem = val//16 
      rem = rem + 55 if rem > 9 else rem + 48
      hex_val += chr(rem)
      rem = val%16 
      rem = rem + 55 if rem > 9 else rem + 48
      hex_val += chr(rem)
    return hex_val

  def rev_hex_conv(self, text):
    hex_val = ''
    for i in range(0,16,2):
      val = ord(text[i])
      val = val - 65 + 10 if val > 57 else val - 48
      val *= 16
      val1 = ord(text[i+1])
      val1 = val1 - 65 + 10 if val1 > 57 else val1 - 48
      hex_val += chr(val + val1)
    return hex_val

  def pad(self,plainText):
    add_x = 8 - len(plainText) % 8
    if add_x == 0: 
      add_x = 8
    for i in range(0,add_x-1):
      plainText += '_'
    plainText += chr(48 + add_x)
    return plainText
  
  def unpad(self,plainText):
    pad = ord(plainText[-1]) - 48
    plainText = plainText[0:-pad]
    return plainText

  def xor(self, a, b):
    ans = ""
    for i in range(len(a)):
      if a[i] == b[i]:
        ans = ans + "0"
      else:
        ans = ans + "1"
    return ans

  def hex2bin(self, s):
    mp = {'0': "0000",
      '1': "0001",
      '2': "0010",
      '3': "0011",
      '4': "0100",
      '5': "0101",
      '6': "0110",
      '7': "0111",
      '8': "1000",
      '9': "1001",
      'A': "1010",
      'B': "1011",
      'C': "1100",
      'D': "1101",
      'E': "1110",
      'F': "1111"}
    bin = ""
    for i in range(len(s)):
      bin = bin + mp[s[i]]
    return bin

  # Binary to hexadecimal conversion

  def bin2hex(self, s):
    mp = {"0000": '0',
      "0001": '1',
      "0010": '2',
      "0011": '3',
      "0100": '4',
      "0101": '5',
      "0110": '6',
      "0111": '7',
      "1000": '8',
      "1001": '9',
      "1010": 'A',
      "1011": 'B',
      "1100": 'C',
      "1101": 'D',
      "1110": 'E',
      "1111": 'F'}
    hex = ""
    for i in range(0, len(s), 4):
      ch = ""
      ch = ch + s[i]
      ch = ch + s[i + 1]
      ch = ch + s[i + 2]
      ch = ch + s[i + 3]
      hex = hex + mp[ch]

    return hex  
  
  def encrypt_ECB(self,plainText):
    cypherText = '0x'

    # padding 
    plainText = self.pad(plainText)

    # divide into blocks
    for i in range(0,len(plainText),8):
      pt = plainText[i:i+8]
      pt = self.hex_conv(pt)
      pt = self.encrypt(pt,0)
      cypherText += pt
    return cypherText, self.act_key  
  
  def decrypt_ECB(self,cypherText, u_pad=1):
    plainText = ''
    cypherText = cypherText[2:len(cypherText)] 
    # divide into blocks
    for i in range(0,len(cypherText),16):
      ct = cypherText[i:i+16]
      ct = self.encrypt(ct,1)
      ct = self.rev_hex_conv(ct)
      plainText += ct
      print("\n",ct,"\n")  
    
    # unpadding 
    if u_pad == 1:
        plainText = self.unpad(plainText)   
    
    return plainText
  
  def encrypt_cbc(self, plaintext, iv):
        """
        Encrypts `plaintext` using CBC mode and PKCS#7 padding, with the given
        initialization vector (iv).
        """
        assert len(iv) == 8

        plaintext = self.pad(plaintext)

        cypherText = '0x'
        previous = self.hex_conv(iv)
        for i in range(0,len(plaintext),8):
            # CBC mode encrypt: encrypt(plaintext_block XOR previous)
            pt = plaintext[i:i+8]
            pt = self.hex_conv(pt)
            pt = self.bin2hex(self.xor(self.hex2bin(pt), self.hex2bin(previous)))
            block = self.encrypt(pt,0)
            cypherText += block
            previous = block

        return cypherText, self.act_key

  def decrypt_cbc(self, cyphertext, iv, u_pad=1):
      """
      Decrypts `ciphertext` using CBC mode and PKCS#7 padding, with the given
      initialization vector (iv).
      """
      assert len(iv) == 8
      cyphertext = cyphertext[2:len(cyphertext)] 
      plaintext = ''
      previous = self.hex_conv(iv)
      for i in range(0,len(cyphertext),16):
          # CBC mode decrypt: previous XOR decrypt(ciphertext)
          ct = cyphertext[i:i+16]
          dct = self.encrypt(ct,1)
          pt = self.xor(self.hex2bin(previous), self.hex2bin(dct))
          pt = self.bin2hex(pt)
          pt = self.rev_hex_conv(pt)
          plaintext += pt
          previous = ct

      if u_pad == 1:
        plaintext = self.unpad(plaintext)
      return plaintext

  def encrypt_cfb(self, plaintext, iv):
      """
      Encrypts `plaintext` with the given initialization vector (iv).
      """
      assert len(iv) == 8
      plaintext = self.pad(plaintext)
      cyphertext = '0x'
      previous = self.hex_conv(iv)
      for i in range(0,len(plaintext),8):
          # CFB mode encrypt: plaintext_block XOR encrypt(prev_ciphertext)
          pt = plaintext[i:i+8]
          pt = self.hex2bin(self.hex_conv(pt)) 
          dpt = self.encrypt(previous,0)
          ciphertext_block = self.xor(pt, self.hex2bin(dpt))
          ciphertext_block = self.bin2hex(ciphertext_block)
          cyphertext += ciphertext_block
          previous = ciphertext_block

      return cyphertext, self.act_key

  def decrypt_cfb(self, cyphertext, iv, u_pad=1):
      """
      Decrypts `ciphertext` with the given initialization vector (iv).
      """
      assert len(iv) == 8
      cyphertext = cyphertext[2:len(cyphertext)]
      plaintext = ''
      previous = self.hex_conv(iv)
      for i in range(0,len(cyphertext),16):
          # CFB mode decrypt: ciphertext XOR decrypt(prev_ciphertext)
          ct = cyphertext[i:i+16]
          bct = self.encrypt(previous,0)
          pt= self.xor(self.hex2bin(ct), self.hex2bin(bct))
          pt = self.bin2hex(pt)
          pt = self.rev_hex_conv(pt)
          plaintext += pt
          previous = ct

      if u_pad == 1:
        plaintext = self.unpad(plaintext)
      return plaintext

  def encrypt_ofb(self, plaintext, iv):
      """
      Encrypts `plaintext` using OFB mode initialization vector (iv).
      """
      assert len(iv) == 8
      plaintext = self.pad(plaintext)
      cyphertext = '0x'
      previous = self.hex_conv(iv)
      for i in range(0,len(plaintext),8):
          # OFB mode encrypt: plaintext_block XOR encrypt(previous)
          block = self.encrypt(previous,0)
          pt = plaintext[i:i+8]
          pt = self.hex2bin(self.hex_conv(pt))
          ciphertext_block = self.xor(pt, self.hex2bin(block))
          ciphertext_block = self.bin2hex(ciphertext_block)
          cyphertext += ciphertext_block
          previous = block
      return cyphertext, self.act_key

  def decrypt_ofb(self, cyphertext, iv, u_pad=1):
      """
      Decrypts `ciphertext` using OFB mode initialization vector (iv).
      """
      assert len(iv) == 8
      cyphertext = cyphertext[2:len(cyphertext)]
      plaintext = ''
      previous = self.hex_conv(iv)
      for i in range(0,len(cyphertext),16):
          # OFB mode decrypt: ciphertext XOR encrypt(previous)
          ct = cyphertext[i:i+16]
          bct = self.encrypt(previous,0)
          pt= self.xor(self.hex2bin(ct), self.hex2bin(bct))
          pt = self.bin2hex(pt)
          pt = self.rev_hex_conv(pt)
          plaintext += pt
          previous = bct
      if u_pad == 1:
        plaintext = self.unpad(plaintext)
      return plaintext

  def inc_bytes(self,a):
      hexs = '0123456789ABCDEF'
      out = ''
      for i in range(0,16,2):
         if a[i:i+2] == 'FF':
             out += '00'
         else:    
             val = ord(a[i+1])
             val = val - 65 + 10 if val > 57 else val - 48
             if val<15:
                out += a[i]+hexs[val+1]
             else:
                val = ord(a[i])
                val = val - 65 + 10 if val > 57 else val - 48
                out += hexs[val+1] + '0'
      return out


  def encrypt_ctr(self, plaintext, iv):
      """
      Encrypts `plaintext` using CTR mode with the given nounce/IV.
      """
      assert len(iv) == 8
      plaintext = self.pad(plaintext)
      cyphertext = '0x'
      nonce = self.hex_conv(iv)
      for i in range(0,len(plaintext),8):
          # CTR mode encrypt: plaintext_block XOR encrypt(nonce)
          pt = plaintext[i:i+8]
          pt = self.hex2bin(self.hex_conv(pt)) 
          dpt = self.encrypt(nonce,0)
          ciphertext_block = self.xor(pt, self.hex2bin(dpt))
          ciphertext_block = self.bin2hex(ciphertext_block)
          cyphertext += ciphertext_block
          nonce = self.inc_bytes(nonce)
      return cyphertext, self.act_key


  def decrypt_ctr(self, cyphertext, iv, u_pad=1):
      """
      Decrypts `ciphertext` using CTR mode with the given nounce/IV.
      """
      assert len(iv) == 8
      cyphertext = cyphertext[2:len(cyphertext)]
      plaintext = ''
      nonce = self.hex_conv(iv)
      for i in range(0,len(cyphertext),16):
          # CTR mode decrypt: ciphertext XOR encrypt(nonce)
          ct = cyphertext[i:i+16]
          bct = self.encrypt(nonce,0)
          pt= self.xor(self.hex2bin(ct), self.hex2bin(bct))
          pt = self.bin2hex(pt)
          pt = self.rev_hex_conv(pt)
          plaintext += pt
          nonce = self.inc_bytes(nonce)
      if u_pad == 1:
        plaintext = self.unpad(plaintext)
      return plaintext


  def encrypt(self,plaintext,decrypt):
     enc_dec = [[0,1,0],[1,0,1]]
     dec = enc_dec[decrypt]
     if decrypt == 0:
        des1 = DES(self.act_key[0])
        des2 = DES(self.act_key[1])
        des3 = DES(self.act_key[2])
     else:
        des1 = DES(self.act_key[2])
        des2 = DES(self.act_key[1])
        des3 = DES(self.act_key[0])
     ct = des1.encrypt(plaintext,dec[0])
     ct = des2.encrypt(ct,dec[1])
     ct = des3.encrypt(ct,dec[2])
     return ct
     
  
def _201cs105_encrypt_file(file_to_encrypt, cryptogram_file, algorithm):
    if algorithm not in ['DES', 'TDES', 'AES']:
        raise ValueError("Invalid encryption algorithm. Choose from: 'DES', 'TripleDES', 'AES'")
    
    if algorithm in ['DES','TDES']:
      with open(file_to_encrypt, 'r') as f:
          plaintext = f.read()
      
      if algorithm == 'DES':
        key = input("Enter 8 character long key: ")
        algo = DES(key)
      elif algorithm == 'TDES':
        key = input("Enter 24 character long key: ")
        algo = TDES(key) 
      
      mod = input("choose mode(ECB,CBC,CFB,OFB,CTR): ")
      if mod not in ['ECB','CBC','CFB','OFB','CTR']:
        raise ValueError("Invalid encryption algorithm.")
      elif mod != 'ECB':
        iv = input('Enter 8 character long IV: ')

      if mod == 'ECB':
        cyphertext, nkey = algo.encrypt_ECB(plaintext)
        decr_text = algo.decrypt_ECB(cyphertext)
      elif mod == 'CBC':
        cyphertext, nkey = algo.encrypt_cbc(plaintext,iv)
        decr_text = algo.decrypt_cbc(cyphertext,iv)
      elif mod == 'CFB':
        cyphertext, nkey = algo.encrypt_cfb(plaintext,iv)
        decr_text = algo.decrypt_cfb(cyphertext,iv)
      elif mod == 'OFB':
        cyphertext, nkey = algo.encrypt_ofb(plaintext,iv)
        decr_text = algo.decrypt_ofb(cyphertext,iv)
      elif mod == 'CTR':
        cyphertext, nkey = algo.encrypt_ctr(plaintext,iv)
        decr_text = algo.decrypt_ctr(cyphertext,iv)

      with open(cryptogram_file, 'w') as f:
          f.write(cyphertext)
      with open("validate.txt", 'w') as f:
          f.write(decr_text)
      print("for validation of encryption & decryption check validate.txt")
      print(nkey)
    else:
      with open(file_to_encrypt, 'rb') as f:
        plaintext = f.read()
    
      key = input("Enter 16 character long key: ")
      algo = AES(key.encode())
      
      mod = input("choose mode(ECB,CBC,CFB,OFB,CTR): ")
      if mod not in ['ECB','CBC','CFB','OFB','CTR']:
        raise ValueError("Invalid encryption algorithm.")
      elif mod != 'ECB':
        iv = input('Enter 16 character long IV: ')
        iv = iv.encode()

      if mod == 'ECB':
        cyphertext = algo.encrypt_ecb(plaintext)
        decr_text = algo.decrypt_ecb(cyphertext)
      elif mod == 'CBC':
        cyphertext = algo.encrypt_cbc(plaintext,iv)
        decr_text = algo.decrypt_cbc(cyphertext,iv)
      elif mod == 'CFB':
        cyphertext = algo.encrypt_cfb(plaintext,iv)
        decr_text = algo.decrypt_cfb(cyphertext,iv)
      elif mod == 'OFB':
        cyphertext = algo.encrypt_ofb(plaintext,iv)
        decr_text = algo.decrypt_ofb(cyphertext,iv)
      elif mod == 'CTR':
        cyphertext = algo.encrypt_ctr(plaintext,iv)
        decr_text = algo.decrypt_ctr(cyphertext,iv)

      with open(cryptogram_file, 'wb') as f:
          f.write(cyphertext)
      with open("validate.txt", 'wb') as f:
          f.write(decr_text)
      print("for validation of encryption & decryption check validate.txt")



_201cs105_file_to_encrypt = input("Enter the name of the file to encrypt: ")
_201cs105_cryptogram_file = input("Enter the name of the file to store the cryptogram: ")
_201cs105_algorithm = input("Enter the name of the encryption algorithm: ")
# _201cs105_encrypt_file("input.txt", "enc.txt", _201cs105_algorithm)
_201cs105_encrypt_file(_201cs105_file_to_encrypt, _201cs105_cryptogram_file, _201cs105_algorithm)

   