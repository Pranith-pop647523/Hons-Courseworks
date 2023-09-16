
#include <infos/util/map.h>
#include <arch/x86/pio.h>
#include <infos/util/string.h>
#include <infos/drivers/ata/page-cache.h>
#include <infos/kernel/log.h>
#include <infos/kernel/kernel.h>

using namespace infos::util;
using namespace infos::kernel;


    LRUCache::LRUCache(int capacity) : _capacity(capacity),_size(0),_head(nullptr),_tail(nullptr){}

	/**
	 * Returns true if block is present within the cache and copies data to the address pointed to by the buffer pointer
	 * @param buffer pointer to address of buffer.
	 * @param offset used as a key for the map in the cache.
	 * @return Returns true if block is present within the cache 
	 */

    bool LRUCache::contains(int offset, void* buffer) {

        Node* node;
        if (_cache.try_get_value(offset, node) && node->isValid) { // Cache hit if true
            memcpy(buffer, node->data, 512);
            makeMostRecent(node);     // Move the node to the front making it the most recently used node
            return true;
            
        } 

        return false;

    }
        

	/**
	 * Essentially creates a key, value pair where the key is the offset and value is the pointer to the node which contains the data at the buffer address
     * We copy the data at the buffer to this node so that we can use it later in the cache with O(1) lookup
	 * @param buffer pointer to address of buffer.
	 * @param offset used as a key for the map in the cache.
	 * @return void
	 */        

        
    void LRUCache::addLRU(int offset, void* buffer)      // implies Cache miss
        
        {
           

            //add a new node to the dll 

            Node* newNode = new Node();
            newNode->offset = offset;
            newNode->isValid = true;
            newNode->prev = nullptr;  
            newNode->next = nullptr;
            newNode->data = new uint8_t[512];



            memcpy(newNode->data, buffer, 512); 

            if (_size == _capacity) {
                // Remove the least recently used node based on LRU policy
                Node* lastNode = _tail;
                lastNode->isValid = false;
                sys.mm().objalloc().free(lastNode->data); //frees data stored in node
                _tail = _tail->prev;
              
                if (_tail) {
                    _tail->next = nullptr;
                } else {
                    _head = nullptr;
                }
  
            } else {
                _size++;
            }

            // Insert the new node at the head
            newNode->next = _head;
            if (_head) {
                _head->prev = newNode;
            } else {
                _tail = newNode;
            }
            _head = newNode;

            // Update the cache
            _cache.add(offset,newNode);
    

        }
 	/**
	 * Moves the node to the head of the list , essentialy making it the most recently used node in the dll
	 * @param node
	 * @return void
	 */           


    void LRUCache::makeMostRecent(Node* node) {
        if (node == _head) {
            return;
        }

        // Remove node from the current position
        if (node->prev) {
            node->prev->next = node->next;
        }
        if (node->next) {
            node->next->prev = node->prev;
        } else {
            _tail = node->prev;
        }

        // Move the node to the head
        node->prev = nullptr;
        node->next = _head;
        _head->prev = node;
        _head = node;
    }
	/**
	 * Essentially creates a key, value pair where the key is the offset and value is the pointer to the node which contains the data at the buffer address
     * We copy the data at the buffer to this node so that we can use it later in the cache with O(1) lookup. uses an mru eviction policy if capacity is exceeded
	 * @param buffer pointer to address of buffer.
	 * @param offset used as a key for the map in the cache.
	 * @return void
	 */        
    
void LRUCache::addMRU(int offset, void* buffer)      // implies Cache miss
        
        {
           

            //add a new node to the dll 

            Node* newNode = new Node();
            newNode->offset = offset;
            newNode->isValid = true;
            newNode->prev = nullptr;  
            newNode->next = nullptr;
            newNode->data = new uint8_t[512];


            memcpy(newNode->data, buffer, 512); 

            if (_size == _capacity) {
                // Remove the most recently used node based on MRU policy
                Node* firstNode = _head;
                firstNode->isValid = false;
                sys.mm().objalloc().free(firstNode->data); //frees data stored in node
                _head = _head->next;
         
                if (_head) {
                    _head->prev = nullptr;
                } else {
                    _tail = nullptr;
                }
            
            } else {
                _size++;
            }
            

            // Insert the new node at the head
            newNode->next = _head;
            if (_head) {
                _head->prev = newNode;
            } else {
                _tail = newNode;
            }
            _head = newNode;

            // Update the cache
            _cache.add(offset,newNode);
    

        }

