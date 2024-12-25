#pragma once
#include <vector>

class MemoryPool {
private:
    static constexpr size_t BLOCK_SIZE = 4096;
    static constexpr size_t MAX_ALLOC = 256;
    
    struct Block {
        char data[BLOCK_SIZE];
        size_t used = 0;
        Block* next = nullptr;
    };
    
    Block* current_block = nullptr;
    std::vector<Block*> blocks;
    
public:
    MemoryPool() {
        current_block = new Block();
        blocks.push_back(current_block);
    }
    
    ~MemoryPool() {
        for (auto block : blocks) delete block;
    }
    
    void* allocate(size_t size) {
        if (size > MAX_ALLOC) return ::operator new(size);
        
        size = (size + 7) & ~7; // Align to 8 bytes
        
        if (current_block->used + size > BLOCK_SIZE) {
            auto new_block = new Block();
            blocks.push_back(new_block);
            current_block = new_block;
        }
        
        void* result = current_block->data + current_block->used;
        current_block->used += size;
        return result;
    }
    
    void deallocate(void* ptr, size_t size) {
        if (size > MAX_ALLOC) ::operator delete(ptr);
    }
};