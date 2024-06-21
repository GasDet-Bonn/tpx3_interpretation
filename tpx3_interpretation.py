from tqdm import tqdm
import numpy as np
import tables as tb
import sys
from multiprocessing import Pool

from basil.utils.BitLogic import BitLogic

class AssignmentError(Exception):
    def __init__(self, message):
        self.message = message

# Initialize the look up tables for decoding
_lfsr_4_lut = np.zeros((2 ** 4), dtype=np.uint16)
_lfsr_10_lut = np.zeros((2 ** 10), dtype=np.uint16)
_lfsr_14_lut = np.zeros((2 ** 14), dtype=np.uint16)
_gray_14_lut = np.zeros((2 ** 14), dtype=np.uint16)

# Fill the 4-bit LFSR look up table
lfsr = BitLogic(4)
lfsr[3:0] = 0xF
dummy = 0
for i in range(2**4):
    _lfsr_4_lut[BitLogic.tovalue(lfsr)] = i
    dummy = lfsr[3]
    lfsr[3] = lfsr[2]
    lfsr[2] = lfsr[1]
    lfsr[1] = lfsr[0]
    lfsr[0] = lfsr[3] ^ dummy
_lfsr_4_lut[2 ** 4 - 1] = 0

# Fill the 10-bit LFSR look up table
lfsr = BitLogic(10)
lfsr[7:0] = 0xFF
lfsr[9:8] = 0b11
dummy = 0
for i in range(2 ** 10):
    _lfsr_10_lut[BitLogic.tovalue(lfsr)] = i
    dummy = lfsr[9]
    lfsr[9] = lfsr[8]
    lfsr[8] = lfsr[7]
    lfsr[7] = lfsr[6]
    lfsr[6] = lfsr[5]
    lfsr[5] = lfsr[4]
    lfsr[4] = lfsr[3]
    lfsr[3] = lfsr[2]
    lfsr[2] = lfsr[1]
    lfsr[1] = lfsr[0]
    lfsr[0] = lfsr[7] ^ dummy
_lfsr_10_lut[2 ** 10 - 1] = 0

# Fill the 14-bit LFSR look up table
lfsr = BitLogic(14)
lfsr[7:0] = 0xFF
lfsr[13:8] = 63
dummy = 0
for i in range(2**14):
    _lfsr_14_lut[BitLogic.tovalue(lfsr)] = i
    dummy = lfsr[13]
    lfsr[13] = lfsr[12]
    lfsr[12] = lfsr[11]
    lfsr[11] = lfsr[10]
    lfsr[10] = lfsr[9]
    lfsr[9] = lfsr[8]
    lfsr[8] = lfsr[7]
    lfsr[7] = lfsr[6]
    lfsr[6] = lfsr[5]
    lfsr[5] = lfsr[4]
    lfsr[4] = lfsr[3]
    lfsr[3] = lfsr[2]
    lfsr[2] = lfsr[1]
    lfsr[1] = lfsr[0]
    lfsr[0] = lfsr[2] ^ dummy ^ lfsr[12] ^ lfsr[13]
_lfsr_14_lut[2 ** 14 - 1] = 0

# Fill the 14-bit gray look up table
for j in range(2**14):
    encoded_value = BitLogic(14) #48
    encoded_value[13:0]=j #47
    gray_decrypt_v = BitLogic(14) #48
    gray_decrypt_v[13]=encoded_value[13] #47
    for i in range (12, -1, -1): #46
        gray_decrypt_v[i]=gray_decrypt_v[i+1]^encoded_value[i]
    _gray_14_lut[j] = gray_decrypt_v.tovalue()

def interpret_data(args):
    input_filename, raw_indices, op_mode, vco, scan_id, scan_param_id, chunk_start_time = args
    with tb.open_file(input_filename, 'r') as h5_file_in:
        raw_data = h5_file_in.root.raw_data[raw_indices]

    # Based on the headers, filter for hit words and create a list of these words and a list of their indices
    hit_filter = np.where(np.right_shift(np.bitwise_and(raw_data, 0xf0000000), 28) != 0b0101)
    hits = raw_data[hit_filter]
    hits_indices = raw_indices[hit_filter]

    # Only in "DataTake" the ToA extensions are active
    if scan_id == 'DataTake':
        # Based on the headers, filter for ToA extension words and create a list of these words and a list of their indices
        timestamp_map = np.right_shift(np.bitwise_and(raw_data, 0xf0000000), 28) == 0b0101
        timestamp_filter = np.where(timestamp_map == True)
        timestamps = raw_data[timestamp_filter]
        timestamps_indices = raw_indices[timestamp_filter]

        # Split the lists in separate lists for word 0 and word 1 (based on header)
        timestamps_0_filter = np.where(np.right_shift(np.bitwise_and(timestamps, 0x3000000), 24) == 0b01)
        timestamps_1_filter = np.where(np.right_shift(np.bitwise_and(timestamps, 0x3000000), 24) == 0b10)
        timestamps_0 = timestamps[timestamps_0_filter].astype(np.uint64)
        timestamps_1 = timestamps[timestamps_1_filter].astype(np.uint64)
        timestamps_0_indices = timestamps_indices[timestamps_0_filter]
        timestamps_1_indices = timestamps_indices[timestamps_1_filter]

        # Combine the timestamp bits of word 0 and word 1 to the full 48-bit ToA extension
        if len(timestamps_1) == len(timestamps_0):
            full_timestamps = np.left_shift(np.bitwise_and(timestamps_1, 0xffffff), 24) + np.bitwise_and(timestamps_0, 0xfff000)
            full_timestamps_indices = timestamps_0_indices
        elif len(timestamps_1) + 1 == len(timestamps_0):
            full_timestamps = np.left_shift(np.bitwise_and(timestamps_1, 0xffffff), 24) + np.bitwise_and(timestamps_0[:-1], 0xfff000)
            full_timestamps_indices = timestamps_0_indices[:-1]
        elif len(timestamps_1) == len(timestamps_0) + 1:
            full_timestamps = np.left_shift(np.bitwise_and(timestamps_1[:-1], 0xffffff), 24) + np.bitwise_and(timestamps_0, 0xfff000)
            full_timestamps_indices = timestamps_0_indices
        else:
            try:
                raise AssignmentError("Words for timestamps do not match - assignment not possible")
            except AssignmentError as error:
                print(error)
                print("Stop interpretation")
                quit()
    
    raw_data = None

    # Split the hit word list up into lists of words for the individual chip links
    # First: create the filer for this
    link0_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00000000)
    link1_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00000010)
    link2_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00000100)
    link3_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00000110)
    link4_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00001000)
    link5_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00001010)
    link6_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00001100)
    link7_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00001110)

    # Second: Filter the words
    link0_words = hits[link0_hits_filter]
    link1_words = hits[link1_hits_filter]
    link2_words = hits[link2_hits_filter]
    link3_words = hits[link3_hits_filter]
    link4_words = hits[link4_hits_filter]
    link5_words = hits[link5_hits_filter]
    link6_words = hits[link6_hits_filter]
    link7_words = hits[link7_hits_filter]
    hits = None

    # Third: Apply the filter also to the indices
    link0_words_indices = hits_indices[link0_hits_filter]
    link1_words_indices = hits_indices[link1_hits_filter]
    link2_words_indices = hits_indices[link2_hits_filter]
    link3_words_indices = hits_indices[link3_hits_filter]
    link4_words_indices = hits_indices[link4_hits_filter]
    link5_words_indices = hits_indices[link5_hits_filter]
    link6_words_indices = hits_indices[link6_hits_filter]
    link7_words_indices = hits_indices[link7_hits_filter]

    # Split the hit list for the links up into separate lists with word 0 and word 1
    # First: create the filter
    link0_words0_filter = np.where(np.right_shift(np.bitwise_and(link0_words, 0x1000000), 24) == 0b0)
    link0_words1_filter = np.where(np.right_shift(np.bitwise_and(link0_words, 0x1000000), 24) == 0b1)
    link1_words0_filter = np.where(np.right_shift(np.bitwise_and(link1_words, 0x1000000), 24) == 0b0)
    link1_words1_filter = np.where(np.right_shift(np.bitwise_and(link1_words, 0x1000000), 24) == 0b1)
    link2_words0_filter = np.where(np.right_shift(np.bitwise_and(link2_words, 0x1000000), 24) == 0b0)
    link2_words1_filter = np.where(np.right_shift(np.bitwise_and(link2_words, 0x1000000), 24) == 0b1)
    link3_words0_filter = np.where(np.right_shift(np.bitwise_and(link3_words, 0x1000000), 24) == 0b0)
    link3_words1_filter = np.where(np.right_shift(np.bitwise_and(link3_words, 0x1000000), 24) == 0b1)
    link4_words0_filter = np.where(np.right_shift(np.bitwise_and(link4_words, 0x1000000), 24) == 0b0)
    link4_words1_filter = np.where(np.right_shift(np.bitwise_and(link4_words, 0x1000000), 24) == 0b1)
    link5_words0_filter = np.where(np.right_shift(np.bitwise_and(link5_words, 0x1000000), 24) == 0b0)
    link5_words1_filter = np.where(np.right_shift(np.bitwise_and(link5_words, 0x1000000), 24) == 0b1)
    link6_words0_filter = np.where(np.right_shift(np.bitwise_and(link6_words, 0x1000000), 24) == 0b0)
    link6_words1_filter = np.where(np.right_shift(np.bitwise_and(link6_words, 0x1000000), 24) == 0b1)
    link7_words0_filter = np.where(np.right_shift(np.bitwise_and(link7_words, 0x1000000), 24) == 0b0)
    link7_words1_filter = np.where(np.right_shift(np.bitwise_and(link7_words, 0x1000000), 24) == 0b1)

    # Second: Create the new lists and remove the headers
    link0_words0 = np.right_shift(np.bitwise_and(link0_words[link0_words0_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link0_words1 = np.right_shift(np.bitwise_and(link0_words[link0_words1_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link1_words0 = np.right_shift(np.bitwise_and(link1_words[link1_words0_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link1_words1 = np.right_shift(np.bitwise_and(link1_words[link1_words1_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link2_words0 = np.right_shift(np.bitwise_and(link2_words[link2_words0_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link2_words1 = np.right_shift(np.bitwise_and(link2_words[link2_words1_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link3_words0 = np.right_shift(np.bitwise_and(link3_words[link3_words0_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link3_words1 = np.right_shift(np.bitwise_and(link3_words[link3_words1_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link4_words0 = np.right_shift(np.bitwise_and(link4_words[link4_words0_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link4_words1 = np.right_shift(np.bitwise_and(link4_words[link4_words1_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link5_words0 = np.right_shift(np.bitwise_and(link5_words[link5_words0_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link5_words1 = np.right_shift(np.bitwise_and(link5_words[link5_words1_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link6_words0 = np.right_shift(np.bitwise_and(link6_words[link6_words0_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link6_words1 = np.right_shift(np.bitwise_and(link6_words[link6_words1_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link7_words0 = np.right_shift(np.bitwise_and(link7_words[link7_words0_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)
    link7_words1 = np.right_shift(np.bitwise_and(link7_words[link7_words1_filter], 0xffffff).view('>u4'), 8).astype(np.uint64)

    # Third: Combine word 0 and word 1 to the full 48-bit hit word
    # Fourth: Apply the filter to the indices - Use the index of word 0 als index for the full hit
    if len(link0_words0) == len(link0_words1):
        link0_hits = np.left_shift(link0_words0, 24) + link0_words1
        link0_hits_indices = link0_words_indices[link0_words0_filter]
    elif len(link0_words0) + 1 == len(link0_words1):
        link0_hits = np.left_shift(link0_words0, 24) + link0_words1[:-1]
        link0_hits_indices = link0_words_indices[link0_words0_filter]
    elif len(link0_words0) == len(link0_words1) + 1:
        link0_hits = np.left_shift(link0_words0[:-1], 24) + link0_words1
        link0_hits_indices = link0_words_indices[link0_words0_filter[:-1]]
    else:
        try:
            raise AssignmentError("Words on link 0 do not match - assignment not possible")
        except AssignmentError as error:
            print(error)
            print("Stop interpretation")
            quit()

    if len(link1_words0) == len(link1_words1):
        link1_hits = np.left_shift(link1_words0, 24) + link1_words1
        link1_hits_indices = link1_words_indices[link1_words0_filter]
    elif len(link1_words0) + 1 == len(link1_words1):
        link1_hits = np.left_shift(link1_words0, 24) + link1_words1[:-1]
        link1_hits_indices = link1_words_indices[link1_words0_filter]
    elif len(link1_words0) == len(link1_words1) + 1:
        link1_hits = np.left_shift(link1_words0[:-1], 24) + link1_words1
        link1_hits_indices = link1_words_indices[link1_words0_filter[:-1]]
    else:
        try:
            raise AssignmentError("Words on link 1 do not match - assignment not possible")
        except AssignmentError as error:
            print(error)
            print("Stop interpretation")
            quit()

    if len(link2_words0) == len(link2_words1):
        link2_hits = np.left_shift(link2_words0, 24) + link2_words1
        link2_hits_indices = link2_words_indices[link2_words0_filter]
    elif len(link2_words0) + 1 == len(link2_words1):
        link2_hits = np.left_shift(link2_words0, 24) + link2_words1[:-1]
        link2_hits_indices = link2_words_indices[link2_words0_filter]
    elif len(link2_words0) == len(link2_words1) + 1:
        link2_hits = np.left_shift(link2_words0[:-1], 24) + link2_words1
        link2_hits_indices = link2_words_indices[link2_words0_filter[:-1]]
    else:
        try:
            raise AssignmentError("Words on link 2 do not match - assignment not possible")
        except AssignmentError as error:
            print(error)
            print("Stop interpretation")
            quit()

    if len(link3_words0) == len(link3_words1):
        link3_hits = np.left_shift(link3_words0, 24) + link3_words1
        link3_hits_indices = link3_words_indices[link3_words0_filter]
    elif len(link3_words0) + 1 == len(link3_words1):
        link3_hits = np.left_shift(link3_words0, 24) + link3_words1[:-1]
        link3_hits_indices = link3_words_indices[link3_words0_filter]
    elif len(link3_words0) == len(link3_words1) + 1:
        link3_hits = np.left_shift(link3_words0[:-1], 24) + link3_words1
        link3_hits_indices = link3_words_indices[link3_words0_filter[:-1]]
    else:
        try:
            raise AssignmentError("Words on link 3 do not match - assignment not possible")
        except AssignmentError as error:
            print(error)
            print("Stop interpretation")
            quit()

    if len(link4_words0) == len(link4_words1):
        link4_hits = np.left_shift(link4_words0, 24) + link4_words1
        link4_hits_indices = link4_words_indices[link4_words0_filter]
    elif len(link4_words0) + 1 == len(link4_words1):
        link4_hits = np.left_shift(link4_words0, 24) + link4_words1[:-1]
        link4_hits_indices = link4_words_indices[link4_words0_filter]
    elif len(link4_words0) == len(link4_words1) + 1:
        link4_hits = np.left_shift(link4_words0[:-1], 24) + link4_words1
        link4_hits_indices = link4_words_indices[link4_words0_filter[:-1]]
    else:
        try:
            raise AssignmentError("Words on link 4 do not match - assignment not possible")
        except AssignmentError as error:
            print(error)
            print("Stop interpretation")
            quit()

    if len(link5_words0) == len(link5_words1):
        link5_hits = np.left_shift(link5_words0, 24) + link5_words1
        link5_hits_indices = link5_words_indices[link5_words0_filter]
    elif len(link5_words0) + 1 == len(link5_words1):
        link5_hits = np.left_shift(link5_words0, 24) + link5_words1[:-1]
        link5_hits_indices = link5_words_indices[link5_words0_filter]
    elif len(link5_words0) == len(link5_words1) + 1:
        link5_hits = np.left_shift(link5_words0[:-1], 24) + link5_words1
        link5_hits_indices = link5_words_indices[link5_words0_filter[:-1]]
    else:
        try:
            raise AssignmentError("Words on link 5 do not match - assignment not possible")
        except AssignmentError as error:
            print(error)
            print("Stop interpretation")
            quit()

    if len(link6_words0) == len(link6_words1):
        link6_hits = np.left_shift(link6_words0, 24) + link6_words1
        link6_hits_indices = link6_words_indices[link6_words0_filter]
    elif len(link6_words0) + 1 == len(link6_words1):
        link6_hits = np.left_shift(link6_words0, 24) + link6_words1[:-1]
        link6_hits_indices = link6_words_indices[link6_words0_filter]
    elif len(link6_words0) == len(link6_words1) + 1:
        link6_hits = np.left_shift(link6_words0[:-1], 24) + link6_words1
        link6_hits_indices = link6_words_indices[link6_words0_filter[:-1]]
    else:
        try:
            raise AssignmentError("Words on link 6 do not match - assignment not possible")
        except AssignmentError as error:
            print(error)
            print("Stop interpretation")
            quit()

    if len(link7_words0) == len(link7_words1):
        link7_hits = np.left_shift(link7_words0, 24) + link7_words1
        link7_hits_indices = link7_words_indices[link7_words0_filter]
    elif len(link7_words0) + 1 == len(link7_words1):
        link7_hits = np.left_shift(link7_words0, 24) + link7_words1[:-1]
        link7_hits_indices = link7_words_indices[link7_words0_filter]
    elif len(link7_words0) == len(link7_words1) + 1:
        link7_hits = np.left_shift(link7_words0[:-1], 24) + link7_words1
        link7_hits_indices = link7_words_indices[link7_words0_filter[:-1]]
    else:
        try:
            raise AssignmentError("Words on link 7 do not match - assignment not possible")
        except AssignmentError as error:
            print(error)
            print("Stop interpretation")
            quit()

    # When there are ToA extensions combine them with the hits
    if scan_id == 'DataTake':
        # Based on the indices of hits and ToA extensions combine them: Each hit should get the 
        # extensions with the next lowest index
        link0_hits_extensions_indices = np.searchsorted(full_timestamps_indices, link0_hits_indices)
        link1_hits_extensions_indices = np.searchsorted(full_timestamps_indices, link1_hits_indices)
        link2_hits_extensions_indices = np.searchsorted(full_timestamps_indices, link2_hits_indices)
        link3_hits_extensions_indices = np.searchsorted(full_timestamps_indices, link3_hits_indices)
        link4_hits_extensions_indices = np.searchsorted(full_timestamps_indices, link4_hits_indices)
        link5_hits_extensions_indices = np.searchsorted(full_timestamps_indices, link5_hits_indices)
        link6_hits_extensions_indices = np.searchsorted(full_timestamps_indices, link6_hits_indices)
        link7_hits_extensions_indices = np.searchsorted(full_timestamps_indices, link7_hits_indices)

        link0_hits_extensions_indices = np.maximum(link0_hits_extensions_indices - 1, 0)
        link1_hits_extensions_indices = np.maximum(link1_hits_extensions_indices - 1, 0)
        link2_hits_extensions_indices = np.maximum(link2_hits_extensions_indices - 1, 0)
        link3_hits_extensions_indices = np.maximum(link3_hits_extensions_indices - 1, 0)
        link4_hits_extensions_indices = np.maximum(link4_hits_extensions_indices - 1, 0)
        link5_hits_extensions_indices = np.maximum(link5_hits_extensions_indices - 1, 0)
        link6_hits_extensions_indices = np.maximum(link6_hits_extensions_indices - 1, 0)
        link7_hits_extensions_indices = np.maximum(link7_hits_extensions_indices - 1, 0)

        link0_hits_extensions = full_timestamps[link0_hits_extensions_indices]
        link1_hits_extensions = full_timestamps[link1_hits_extensions_indices]
        link2_hits_extensions = full_timestamps[link2_hits_extensions_indices]
        link3_hits_extensions = full_timestamps[link3_hits_extensions_indices]
        link4_hits_extensions = full_timestamps[link4_hits_extensions_indices]
        link5_hits_extensions = full_timestamps[link5_hits_extensions_indices]
        link6_hits_extensions = full_timestamps[link6_hits_extensions_indices]
        link7_hits_extensions = full_timestamps[link7_hits_extensions_indices]

        # Check if bit 12 and 13 of the ToA and the ToA extension are equal (they should be based on the firmware setting of the extension)
        # For hits which dont fulfill this condition store their indices into a list, so that they can be corrected
        link0_extension_offsets = np.where(np.bitwise_and(link0_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link0_hits, 14), 0x3fff)], 0x3000))[0]
        link1_extension_offsets = np.where(np.bitwise_and(link1_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link1_hits, 14), 0x3fff)], 0x3000))[0]
        link2_extension_offsets = np.where(np.bitwise_and(link2_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link2_hits, 14), 0x3fff)], 0x3000))[0]
        link3_extension_offsets = np.where(np.bitwise_and(link3_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link3_hits, 14), 0x3fff)], 0x3000))[0]
        link4_extension_offsets = np.where(np.bitwise_and(link4_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link4_hits, 14), 0x3fff)], 0x3000))[0]
        link5_extension_offsets = np.where(np.bitwise_and(link5_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link5_hits, 14), 0x3fff)], 0x3000))[0]
        link6_extension_offsets = np.where(np.bitwise_and(link6_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link6_hits, 14), 0x3fff)], 0x3000))[0]
        link7_extension_offsets = np.where(np.bitwise_and(link7_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link7_hits, 14), 0x3fff)], 0x3000))[0]

        # Shift the extension index for hits that dont fulfill the condition by -1
        link0_hits_extensions[link0_extension_offsets] -= 1
        link1_hits_extensions[link1_extension_offsets] -= 1
        link2_hits_extensions[link2_extension_offsets] -= 1
        link3_hits_extensions[link3_extension_offsets] -= 1
        link4_hits_extensions[link4_extension_offsets] -= 1
        link5_hits_extensions[link5_extension_offsets] -= 1
        link6_hits_extensions[link6_extension_offsets] -= 1
        link7_hits_extensions[link7_extension_offsets] -= 1

        # Check again the overlap of the two bits after the correction
        link0_extension_offsets = np.where(np.bitwise_and(link0_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link0_hits, 14), 0x3fff)], 0x3000))[0]
        link1_extension_offsets = np.where(np.bitwise_and(link1_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link1_hits, 14), 0x3fff)], 0x3000))[0]
        link2_extension_offsets = np.where(np.bitwise_and(link2_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link2_hits, 14), 0x3fff)], 0x3000))[0]
        link3_extension_offsets = np.where(np.bitwise_and(link3_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link3_hits, 14), 0x3fff)], 0x3000))[0]
        link4_extension_offsets = np.where(np.bitwise_and(link4_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link4_hits, 14), 0x3fff)], 0x3000))[0]
        link5_extension_offsets = np.where(np.bitwise_and(link5_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link5_hits, 14), 0x3fff)], 0x3000))[0]
        link6_extension_offsets = np.where(np.bitwise_and(link6_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link6_hits, 14), 0x3fff)], 0x3000))[0]
        link7_extension_offsets = np.where(np.bitwise_and(link7_hits_extensions, 0x3000) != np.bitwise_and(_gray_14_lut[np.bitwise_and(np.right_shift(link7_hits, 14), 0x3fff)], 0x3000))[0]

    # Combine the link specific lists dor hits and their indices
    data = np.concatenate((link0_hits, link1_hits, link2_hits, link3_hits, link4_hits, link5_hits, link6_hits, link7_hits))
    data_indices = np.concatenate((link0_hits_indices, link1_hits_indices, link2_hits_indices, link3_hits_indices, link4_hits_indices, link5_hits_indices, link6_hits_indices, link7_hits_indices))
    
    # Sort by the indices
    data_sort = np.argsort(data_indices)

    # Apply the new order based on the indices to get the original order of hits
    data = data[data_sort]
    data_indices = data_indices[data_sort]

    # If there are ToA extensions, combine also the link specific lists and apply the ordering
    if scan_id == 'DataTake':
        extensions = np.concatenate((link0_hits_extensions, link1_hits_extensions, link2_hits_extensions, link3_hits_extensions, link4_hits_extensions, link5_hits_extensions, link6_hits_extensions, link7_hits_extensions))
        extensions = extensions[data_sort]

    # Create a list of chunk indices with the length of the hit list, so that for each hit chunk specific information can be added
    chunk_indices = np.searchsorted(start_indices, data_indices, side='right')
    chunk_indices = np.maximum(chunk_indices - 1, 0)

    # Create lists of the scan pararmeter ids and chunk start times for all hits based on the chunk indices
    data_scan_param_ids = scan_param_id[chunk_indices]
    data_chunk_start_time = chunk_start_time[chunk_indices]
    #data_start_indices = start_indices[chunk_indices]

    # Create a recarray for the hit data
    data_type = {'names': ['data_header', 'header', 'hit_index', 'x',     'y',     'TOA',    'TOT',    'EventCounter', 'HitCounter', 'FTOA',  'scan_param_id', 'chunk_start_time', 'iTOT',   'TOA_Extension', 'TOA_Combined'],
            'formats': ['uint8',       'uint8',  'uint64', 'uint8', 'uint8', 'uint16', 'uint16', 'uint16',       'uint8',      'uint8', 'uint16',        'float',            'uint16', 'uint64',        'uint64']}
    pix_data = np.recarray((data.shape[0]), dtype=data_type)

    # Create some numpy numbers for the data interpretation
    n47 = np.uint64(47)
    n44 = np.uint64(44)
    n28 = np.uint64(28)
    n14 = np.uint(14)
    n4 = np.uint64(4)
    n3ff = np.uint64(0x3ff)
    n3fff = np.uint64(0x3fff)
    nf = np.uint64(0xf)

    # Get the x and y coordinates of the hits based on the 3 Timepix3 coordinates (pixel, super_pixel and eoc)
    pixel = (data >> n28) & np.uint64(0b111)
    super_pixel = (data >> np.uint64(28 + 3)) & np.uint64(0x3f)
    right_col = pixel > 3
    eoc = (data >> np.uint64(28 + 9)) & np.uint64(0x7f)
    pix_data['y'] = (super_pixel * 4) + (pixel - right_col * 4)
    pix_data['x'] = eoc * 2 + right_col * 1

    # Put the headers into the recarray
    pix_data['data_header'] = data >> n47
    pix_data['header'] = data >> n44

    # Add chunk based information to the hits into the recarray
    pix_data['scan_param_id'] = data_scan_param_ids
    pix_data['chunk_start_time'] = data_chunk_start_time

    # Write the original indices of word 0 per hit into the recarray
    pix_data['hit_index'] = data_indices

    # Write HitCounter and FTOA based on the run config ()
    if(vco == False):
        pix_data['HitCounter'] = _lfsr_4_lut[data & nf]
        pix_data['FTOA'] = np.zeros(len(data))
    else:
        pix_data['HitCounter'] = np.zeros(len(data))
        pix_data['FTOA'] = data & nf

    # Based on the run mode write iToT, ToT, ToA, EventCounter, ToA_Extension and ToA_combined
    if op_mode == 0b00:
        pix_data['iTOT'] = np.zeros(len(data))
        pix_data['TOT'] = _lfsr_10_lut[(data >> n4) & n3ff]
        pix_data['TOA'] = _gray_14_lut[(data >> n14) & n3fff]
        pix_data['EventCounter'] = np.zeros(len(data))
        if scan_id == 'DataTake':
            pix_data['TOA_Extension'] = extensions & 0xFFFFFFFFFFFF
            pix_data['TOA_Combined'] = (extensions & 0xFFFFFFFFC000) + pix_data['TOA']
        else:
            pix_data['TOA_Extension'] = np.zeros(len(data))
            pix_data['TOA_Combined'] = np.zeros(len(data))
    elif op_mode == 0b01:
        pix_data['iTOT'] = np.zeros(len(data))
        pix_data['TOT'] = np.zeros(len(data))
        pix_data['TOA'] = _gray_14_lut[(data >> n14) & n3fff]
        pix_data['EventCounter'] = np.zeros(len(data))
        if scan_id == 'DataTake':
            pix_data['TOA_Extension'] = extensions & 0xFFFFFFFFFFFF
            pix_data['TOA_Combined'] = (extensions & 0xFFFFFFFFC000) + pix_data['TOA']
        else:
            pix_data['TOA_Extension'] = np.zeros(len(data))
            pix_data['TOA_Combined'] = np.zeros(len(data))
    else:
        pix_data['iTOT'] = _lfsr_14_lut[(data >> n14) & n3fff]
        pix_data['EventCounter'] = _lfsr_10_lut[(data >> n4) & n3ff]
        pix_data['TOT'] = np.zeros(len(data))
        pix_data['TOA'] = np.zeros(len(data))
        if scan_id == 'DataTake':
            pix_data['TOA_Extension'] = extensions & 0xFFFFFFFFFFFF
            pix_data['TOA_Combined'] = (extensions & 0xFFFFFFFFC000) + pix_data['TOA']
        else:
            pix_data['TOA_Extension'] = np.zeros(len(data))
            pix_data['TOA_Combined'] = np.zeros(len(data))

    #print("Order data by timestamp")
    pix_data = pix_data[pix_data['TOA_Combined'].argsort()]

    return pix_data

def save_data(h5_filename_in, h5_filename_out, pix_data):
    # Open the output file
    print("Save data to output file")
    with tb.open_file(h5_filename_out, 'w') as h5_file_out:    
        # If the interpreted node is already there remove it
        try:
            h5_file_out.remove_node(h5_file_out.root.interpreted, recursive=True)
            print("Node interpreted already there")
        except:
            pass

        # Create the groups and their attributes
        interpreted = h5_file_out.create_group(h5_file_out.root, 'interpreted', 'interpreted')
        h5_file_out.root.interpreted._v_attrs['TimepixVersion'] = np.array('Timepix3'.encode("ascii"), dtype=str)
        h5_file_out.root.interpreted._v_attrs['centerChip'] = np.array([0])
        h5_file_out.root.interpreted._v_attrs['runFolderKind'] = np.array('rfUnknown'.encode("ascii"), dtype=str)
        h5_file_out.root.interpreted._v_attrs['runType'] = np.array('rfXrayFinger'.encode("ascii"), dtype=str)
        run_0 = h5_file_out.create_group(h5_file_out.root.interpreted, 'run_0', 'run_0')
        h5_file_out.root.interpreted.run_0._v_attrs['BadBatchCount'] = np.array([0])
        h5_file_out.root.interpreted.run_0._v_attrs['BadSliceCount'] = np.array([0])
        h5_file_out.root.interpreted.run_0._v_attrs['batchSize'] = np.array([100000000])
        h5_file_out.root.interpreted.run_0._v_attrs['numChips'] = np.array([1])

        # Create a table with the interpreted data
        h5_file_out.create_table(h5_file_out.root.interpreted.run_0, 'hit_data', pix_data, filters=tb.Filters(complib='zlib', complevel=2))

        # Copy the chip configuration from the input file to the output file
        h5_file_out.create_group(h5_file_out.root.interpreted.run_0, 'configuration', 'Configuration')
        with tb.open_file(h5_filename_in, 'r+') as h5_file_in:
            h5_file_in.copy_children(h5_file_in.root.configuration, h5_file_out.root.interpreted.run_0.configuration)

if len(sys.argv) != 3:
    print("Please enter the data paths of the input and the output file (python tpx3_interpretation.py <input path> <output path>)")

else:
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    if not input_filename.endswith('.h5'):
        print("Please choose a valid input file")
    if not output_filename.endswith('.h5'):
        print("Please choose a valid output file")

    print("Start interpretation of data ", input_filename)

    with tb.open_file(input_filename, 'r') as h5_file_in:
        # Read the meta data and the chip configuration from the hdf5 file
        meta_data = h5_file_in.root.meta_data[:]
        run_config = h5_file_in.root.configuration.run_config[:]
        general_config = h5_file_in.root.configuration.generalConfig[:]
        op_mode = [row[1] for row in general_config if row[0]==b'Op_mode'][0]
        vco = [row[1] for row in general_config if row[0]==b'Fast_Io_en'][0]
        scan_id = [row[1] for row in run_config if row[0]==b'scan_id'][0].decode()

        # Read the data onto arrays
        meta_data_tmp = meta_data[0:]
        discard_errors = meta_data_tmp['discard_error']
        decode_errors = meta_data_tmp['decode_error']
        errors = discard_errors + decode_errors
        start_indices = meta_data_tmp['index_start']
        stop_indices = meta_data_tmp['index_stop']
        indices = np.array([np.arange(start, stop, 1, dtype=int) for start, stop in zip(start_indices, stop_indices)], dtype=object)
        scan_param_id = meta_data_tmp['scan_param_id']
        chunk_start_time = meta_data_tmp['timestamp_start']
        discarded_packages = 0

        # Get the chunks and the indices of data packages after chunks with errors
        chunks_after_errors = np.where(errors != 0)[0] + 1
        if len(chunks_after_errors) > 0:
            if chunks_after_errors[-1] > (len(indices) - 1):
                chunks_after_errors = chunks_after_errors[:-1]
            #indices_after_errors = indices[chunks_after_errors]
        
        print("Correct data")
        
        i = 0
        for chunk_indices in tqdm(indices, desc="Chunk"):
            # For chunks with errors do noting as they are anyway discarded
            if errors[i] != 0:
                i += 1
                continue
            # Ignore chunks without data
            if len(chunk_indices) == 0:
                i += 1
                continue

            # Based on the headers, filter for hit words and create a list of these words and a list of their indices
            current_raw_data = h5_file_in.root.raw_data[chunk_indices]
            hit_filter = np.where(np.right_shift(np.bitwise_and(current_raw_data, 0xf0000000), 28) != 0b0101)
            hits = current_raw_data[hit_filter]
            hits_indices = chunk_indices[hit_filter]
            hit_filter = None

            # Only in "DataTake" the ToA extensions are active
            if scan_id == 'DataTake':
                # Based on the headers, filter for ToA extension words and create a list of these words and a list of their indices
                timestamp_map = np.right_shift(np.bitwise_and(current_raw_data, 0xf0000000), 28) == 0b0101
                timestamp_filter = np.where(timestamp_map == True)
                timestamps = current_raw_data[timestamp_filter]
                timestamps_indices = chunk_indices[timestamp_filter]

                # Split the lists in separate lists for word 0 and word 1 (based on header)
                timestamps_0_filter = np.where(np.right_shift(np.bitwise_and(timestamps, 0x3000000), 24) == 0b01)[0]
                timestamps_1_filter = np.where(np.right_shift(np.bitwise_and(timestamps, 0x3000000), 24) == 0b10)[0]
                timestamps = None
                timestamp_filter = None
            current_raw_data = None

            # Split the hit word list up into lists of words for the individual chip links
            # First: create the filer for this
            link0_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00000000)
            link1_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00000010)
            link2_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00000100)
            link3_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00000110)
            link4_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00001000)
            link5_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00001010)
            link6_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00001100)
            link7_hits_filter = np.where(np.right_shift(np.bitwise_and(hits, 0xfe000000), 24) == 0b00001110)

            # Second: Filter the words
            link0_words = hits[link0_hits_filter]
            link1_words = hits[link1_hits_filter]
            link2_words = hits[link2_hits_filter]
            link3_words = hits[link3_hits_filter]
            link4_words = hits[link4_hits_filter]
            link5_words = hits[link5_hits_filter]
            link6_words = hits[link6_hits_filter]
            link7_words = hits[link7_hits_filter]

            # Third: Apply the filter also to the indices
            link0_words_indices = hits_indices[link0_hits_filter]
            link1_words_indices = hits_indices[link1_hits_filter]
            link2_words_indices = hits_indices[link2_hits_filter]
            link3_words_indices = hits_indices[link3_hits_filter]
            link4_words_indices = hits_indices[link4_hits_filter]
            link5_words_indices = hits_indices[link5_hits_filter]
            link6_words_indices = hits_indices[link6_hits_filter]
            link7_words_indices = hits_indices[link7_hits_filter]
            hits = None
            hits_indices = None

            # Split the hit list for the links up into separate lists with word 0 and word 1
            # First: create the filter
            link0_words0_filter = np.where(np.right_shift(np.bitwise_and(link0_words, 0x1000000), 24) == 0b0)[0]
            link0_words1_filter = np.where(np.right_shift(np.bitwise_and(link0_words, 0x1000000), 24) == 0b1)[0]
            link1_words0_filter = np.where(np.right_shift(np.bitwise_and(link1_words, 0x1000000), 24) == 0b0)[0]
            link1_words1_filter = np.where(np.right_shift(np.bitwise_and(link1_words, 0x1000000), 24) == 0b1)[0]
            link2_words0_filter = np.where(np.right_shift(np.bitwise_and(link2_words, 0x1000000), 24) == 0b0)[0]
            link2_words1_filter = np.where(np.right_shift(np.bitwise_and(link2_words, 0x1000000), 24) == 0b1)[0]
            link3_words0_filter = np.where(np.right_shift(np.bitwise_and(link3_words, 0x1000000), 24) == 0b0)[0]
            link3_words1_filter = np.where(np.right_shift(np.bitwise_and(link3_words, 0x1000000), 24) == 0b1)[0]
            link4_words0_filter = np.where(np.right_shift(np.bitwise_and(link4_words, 0x1000000), 24) == 0b0)[0]
            link4_words1_filter = np.where(np.right_shift(np.bitwise_and(link4_words, 0x1000000), 24) == 0b1)[0]
            link5_words0_filter = np.where(np.right_shift(np.bitwise_and(link5_words, 0x1000000), 24) == 0b0)[0]
            link5_words1_filter = np.where(np.right_shift(np.bitwise_and(link5_words, 0x1000000), 24) == 0b1)[0]
            link6_words0_filter = np.where(np.right_shift(np.bitwise_and(link6_words, 0x1000000), 24) == 0b0)[0]
            link6_words1_filter = np.where(np.right_shift(np.bitwise_and(link6_words, 0x1000000), 24) == 0b1)[0]
            link7_words0_filter = np.where(np.right_shift(np.bitwise_and(link7_words, 0x1000000), 24) == 0b0)[0]
            link7_words1_filter = np.where(np.right_shift(np.bitwise_and(link7_words, 0x1000000), 24) == 0b1)[0]

            copy_to_next_chunk = []
            move_to_next_chunk = []
            removed_indices = []

            # If a current chunk is after a chunk with errors remove the first word per link if its the wrong one (word 0 instead of 1)
            if i in chunks_after_errors:
                if scan_id == 'DataTake':
                    if timestamps_0_filter[0] < timestamps_1_filter[0]:
                        removed_indices.append(timestamps_indices[0])
                if len(link0_words_indices) > 1:
                    if link0_words0_filter[0] < link0_words1_filter[0]:
                        removed_indices.append(link0_words_indices[0])
                elif len(link0_words_indices) > 0:
                    removed_indices.append(link0_words_indices[0])
                if len(link1_words_indices) > 1:
                    if link1_words0_filter[0] < link1_words1_filter[0]:
                        removed_indices.append(link1_words_indices[0])
                elif len(link1_words_indices) > 0:
                    removed_indices.append(link1_words_indices[0])
                if len(link2_words_indices) > 1:
                    if link2_words0_filter[0] < link2_words1_filter[0]:
                        removed_indices.append(link2_words_indices[0])
                elif len(link2_words_indices) > 0:
                    removed_indices.append(link2_words_indices[0])
                if len(link3_words_indices) > 1:
                    if link3_words0_filter[0] < link3_words1_filter[0]:
                        removed_indices.append(link3_words_indices[0])
                elif len(link3_words_indices) > 0:
                    removed_indices.append(link3_words_indices[0])
                if len(link4_words_indices) > 1:
                    if link4_words0_filter[0] < link4_words1_filter[0]:
                        removed_indices.append(link4_words_indices[0])
                elif len(link4_words_indices) > 0:
                    removed_indices.append(link4_words_indices[0])
                if len(link5_words_indices) > 1:
                    if link5_words0_filter[0] < link5_words1_filter[0]:
                        removed_indices.append(link5_words_indices[0])
                elif len(link5_words_indices) > 0:
                    removed_indices.append(link5_words_indices[0])
                if len(link6_words_indices) > 1:
                    if link6_words0_filter[0] < link6_words1_filter[0]:
                        removed_indices.append(link6_words_indices[0])
                elif len(link6_words_indices) > 0:
                    removed_indices.append(link6_words_indices[0])
                if len(link7_words_indices) > 1:
                    if link7_words0_filter[0] < link7_words1_filter[0]:
                        removed_indices.append(link7_words_indices[0])
                elif len(link7_words_indices) > 0:
                    removed_indices.append(link7_words_indices[0])

            if i + 1 < len(indices):
                # If the difference of 0 and 1 packages is bigger than 0 remove the chunk as there is some error
                if np.abs(len(link0_words0_filter) - len(link0_words1_filter)) > 1:
                    errors[i] += 1
                    chunks_after_errors = np.append(chunks_after_errors, i+1)
                    i += 1
                    continue
                if np.abs(len(link1_words0_filter) - len(link1_words1_filter)) > 1:
                    errors[i] += 1
                    chunks_after_errors = np.append(chunks_after_errors, i+1)
                    i += 1
                    continue
                if np.abs(len(link2_words0_filter) - len(link2_words1_filter)) > 1:
                    errors[i] += 1
                    chunks_after_errors = np.append(chunks_after_errors, i+1)
                    i += 1
                    continue
                if np.abs(len(link3_words0_filter) - len(link3_words1_filter)) > 1:
                    errors[i] += 1
                    chunks_after_errors = np.append(chunks_after_errors, i+1)
                    i += 1
                    continue
                if np.abs(len(link4_words0_filter) - len(link4_words1_filter)) > 1:
                    errors[i] += 1
                    chunks_after_errors = np.append(chunks_after_errors, i+1)
                    i += 1
                    continue
                if np.abs(len(link5_words0_filter) - len(link5_words1_filter)) > 1:
                    errors[i] += 1
                    chunks_after_errors = np.append(chunks_after_errors, i+1)
                    i += 1
                    continue
                if np.abs(len(link6_words0_filter) - len(link6_words1_filter)) > 1:
                    errors[i] += 1
                    chunks_after_errors = np.append(chunks_after_errors, i+1)
                    i += 1
                    continue
                if np.abs(len(link7_words0_filter) - len(link7_words1_filter)) > 1:
                    errors[i] += 1
                    chunks_after_errors = np.append(chunks_after_errors, i+1)
                    i += 1
                    continue

            # If the chunk (per link) ends with a word 1 without fitting word 0 move the word 1 to the next chunk
            if scan_id == 'DataTake':
                if timestamps_0_filter[-1] < timestamps_1_filter[-1]:
                    move_to_next_chunk.append(timestamps_indices[-1])
                    copy_to_next_chunk.append(timestamps_indices[-3:-1])
                else:
                    copy_to_next_chunk.append(timestamps_indices[-2:])
            if len(link0_words_indices) > 1:
                if link0_words0_filter[-1] < link0_words1_filter[-1]:
                    move_to_next_chunk.append(link0_words_indices[-1])
            elif len(link0_words_indices) > 0:
                move_to_next_chunk.append(link0_words_indices[-1])
            if len(link1_words_indices) > 1:
                if link1_words0_filter[-1] < link1_words1_filter[-1]:
                    move_to_next_chunk.append(link1_words_indices[-1])
            elif len(link1_words_indices) > 0:
                move_to_next_chunk.append(link1_words_indices[-1])
            if len(link2_words_indices) > 1:
                if link2_words0_filter[-1] < link2_words1_filter[-1]:
                    move_to_next_chunk.append(link2_words_indices[-1])
            elif len(link2_words_indices) > 0:
                move_to_next_chunk.append(link2_words_indices[-1])
            if len(link3_words_indices) > 1:
                if link3_words0_filter[-1] < link3_words1_filter[-1]:
                    move_to_next_chunk.append(link3_words_indices[-1])
            elif len(link3_words_indices) > 0:
                move_to_next_chunk.append(link3_words_indices[-1])
            if len(link4_words_indices) > 1:
                if link4_words0_filter[-1] < link4_words1_filter[-1]:
                    move_to_next_chunk.append(link4_words_indices[-1])
            elif len(link4_words_indices) > 0:
                move_to_next_chunk.append(link4_words_indices[-1])
            if len(link5_words_indices) > 1:
                if link5_words0_filter[-1] < link5_words1_filter[-1]:
                    move_to_next_chunk.append(link5_words_indices[-1])
            elif len(link5_words_indices) > 0:
                move_to_next_chunk.append(link5_words_indices[-1])
            if len(link6_words_indices) > 1:
                if link6_words0_filter[-1] < link6_words1_filter[-1]:
                    move_to_next_chunk.append(link6_words_indices[-1])
            elif len(link6_words_indices) > 0:
                move_to_next_chunk.append(link6_words_indices[-1])
            if len(link7_words_indices) > 1:
                if link7_words0_filter[-1] < link7_words1_filter[-1]:
                    move_to_next_chunk.append(link7_words_indices[-1])
            elif len(link7_words_indices) > 0:
                move_to_next_chunk.append(link7_words_indices[-1])

            # Create the new indice lists for the current and the next chunk
            new_indices = chunk_indices
            if len(removed_indices) > 0:
                new_indices = np.setdiff1d(new_indices, removed_indices)
                discarded_packages += len(removed_indices)
            if i + 1 < len(indices) and len(move_to_next_chunk) > 0:
                indices[i+1] = np.append(indices[i+1], move_to_next_chunk)
                indices[i+1] = np.sort(indices[i+1])
                new_indices = np.setdiff1d(new_indices, move_to_next_chunk)
            if i + 1 < len(indices) and len(copy_to_next_chunk) > 0:
                indices[i+1] = np.append(indices[i+1], copy_to_next_chunk)
                indices[i+1] = np.sort(indices[i+1])

            indices[i] = new_indices
            i += 1
        
        j = 0
        for chunk_errors in errors:
            if chunk_errors > 0:
                discarded_packages += len(indices[j])
            j += 1
        indices = indices[np.where(errors == 0)[0]]
        print("Discarded packages", discarded_packages, "of", stop_indices[-1], "(", 100. * (discarded_packages / stop_indices[-1]), "%)")

    args = []
    print("Prepare interpretation")
    for index_list in tqdm(indices, desc="Chunk"):
        args.append([input_filename, index_list, op_mode, vco, scan_id, scan_param_id, chunk_start_time])

    print("Interpret data")
    with Pool(4) as pool:
        pix_data = list(tqdm(pool.imap(interpret_data, args, 100), total=len(args), desc="Chunk"))
    pix_data = np.concatenate(pix_data)

    print("Order data by timestamp")
    pix_data = pix_data[pix_data['TOA_Combined'].argsort()]
    save_data(input_filename, output_filename, pix_data)
