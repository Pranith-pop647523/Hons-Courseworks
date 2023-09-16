/*
 * The Buddy Page Allocator
 * SKELETON IMPLEMENTATION TO BE FILLED IN FOR TASK 2
 */

#include <infos/mm/page-allocator.h>
#include <infos/mm/mm.h>
#include <infos/kernel/kernel.h>
#include <infos/kernel/log.h>
#include <infos/util/math.h>
#include <infos/util/printf.h>

using namespace infos::kernel;
using namespace infos::mm;
using namespace infos::util;

#define MAX_ORDER 18
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/**
 * A buddy page allocation algorithm.
 */
class BuddyPageAllocator : public PageAllocatorAlgorithm
{
private:
	/* 
	functions from previous years skeltons have been used
	*/


	/**
	 * Returns the number of pages that comprise a 'block', in a given order.
	 * @param order The order to base the calculation off of.
	 * @return Returns the number of pages in a block, in the order.
	 */
    uint64_t get_block_size(uint64_t order) {
        return 1u << order;
    }

    uint64_t MAX_ORDER_SIZE = get_block_size(MAX_ORDER);


    /**
     * Helper function for sys.mm().pgalloc.pgd_to_pfn() 
     * @param pgd is the pgd we wish to convert to pfn
     * @return the pfn of the given pgd
     */
    pfn_t pgd_to_pfn(PageDescriptor* pgd) {
        return sys.mm().pgalloc().pgd_to_pfn(pgd);
    }

    /**
     * Helper function for sys.mm().pgalloc().pfn_to-pgd() 
     * @param pfn is the pfn we wish to convert to pgd
     * @return the pgd of the given pfn
     */
    PageDescriptor* pfn_to_pgd(pfn_t pfn) {
        return sys.mm().pgalloc().pfn_to_pgd(pfn);
    }

    /**
     * Checks if the pgd is aligned in the block;
     * @param pgd is the pgd that we wish to check the alignment for
     * @param order is the size of the block that we wish to check alignment against
     * @return true if the pgd is aligned within block of size=order, else false
     */
    bool is_aligned(PageDescriptor *pgd, int order) {
        pfn_t pfn = pgd_to_pfn(pgd);
        return (pfn % get_block_size(order) == 0);
    }
    /**
     * Given a page descriptor, and an order, returns the buddy PGD.  The buddy could either be
     * to the left or the right of PGD, in the given order.
     * @param pgd The page descriptor to find the buddy for.
     * @param order The order in which the page descriptor lives.
     * @return Returns the buddy of the given page descriptor, in the given order.
     */
	PageDescriptor *buddy_of(PageDescriptor *pgd, int order)
	{
		// (1) Make sure 'order' is within range
		if (order >= MAX_ORDER) {
			return NULL;
		}

		// (2) Check to make sure that PGD is correctly aligned in the order
		if (!is_aligned(pgd, order)) {
			return NULL;
		}

		// (3) Calculate the page-frame-number of the buddy of this page.
		// * If the PFN is aligned to the next order, then the buddy is the next block in THIS order.
		// * If it's not aligned, then the buddy must be the previous block in THIS order.
		uint64_t buddy_pfn = is_aligned(pgd, order + 1) ?
			sys.mm().pgalloc().pgd_to_pfn(pgd) + get_block_size(order) : 
			sys.mm().pgalloc().pgd_to_pfn(pgd) - get_block_size(order);

		// (4) Return the page descriptor associated with the buddy page-frame-number.
		return sys.mm().pgalloc().pfn_to_pgd(buddy_pfn);
	}

        /**
     * Insert pgd block into the linked list of the given order.
     * Inserts block in between two nodes where the left node is of smaller/equal pgd address than block, and
     * where right node has larger pgd address than block OR is NULL;
     * @param pgd
     * @param order
     */
    PageDescriptor** insert_block(PageDescriptor* pgd, int order) {

        // Find the dbl_ptr to the free_areas array in which the page descriptor should be inserted.
        PageDescriptor **dbl_ptr = &_free_areas[order];
        // Iterate through the linked list to get to the point where prev_free pgd block is the
        // largest block that is smaller than dbl_ptr; stop at the end of the linked list.
        while (*dbl_ptr < pgd and *dbl_ptr != NULL) {
            dbl_ptr = &(*dbl_ptr)->next_free;
        }
        // Insert pgd into the linked list at where dbl_ptr is pointing to:
        pgd->next_free = *dbl_ptr;
        *dbl_ptr = pgd;
        // Return the insert point of dbl_ptr:
        return dbl_ptr;
    }

    /**
     * Removes pgd block of size=order from the free memory linked list.
     * Will only remove the block if it exists!
     * @param pgd is the pgd pointer to the block to be removed
     * @param order is the size of the block
     */
    void remove_block(PageDescriptor* pgd, int order) {

        // Starting from the _free_area array, iterate until the block has been located in the dbl_ptr.
        PageDescriptor **dbl_ptr = &_free_areas[order];
        while (*dbl_ptr < pgd and *dbl_ptr != NULL) {
            dbl_ptr = &(*dbl_ptr)->next_free;
        }

        // Remove the block from the free dbl_ptr.
        *dbl_ptr = pgd->next_free;
        pgd->next_free = NULL;
    }


    /**
     * Given a pointer to a block of free memory in the order "source_order", this function will
     * split the block in half, and insert it into the order below.
     * @param block_pointer A pointer to a pointer containing the beginning of a block of free memory.
     * @param source_order The order in which the block of free memory exists.  Naturally,
     * the split will insert the two new blocks into the order below.
     * @return Returns the left-hand-side of the new block.
     */
    PageDescriptor *split_block(PageDescriptor **block_pointer, int source_order)
    {

        // check alignment in source_order:
        if (!is_aligned(*block_pointer, source_order)) {
            syslog.message(LogLevel::ERROR, "Page descriptor is not aligned");
            return NULL;
        }
        if (source_order == 0) {
            syslog.message(LogLevel::INFO, "Cannot split blocks of order zero");
            return *block_pointer;
        }
        // Find the buddy of the given *block_pointer in the lower order
        PageDescriptor *new_LHS = *block_pointer;
        PageDescriptor *new_RHS = buddy_of(new_LHS, source_order - 1); 

        // Remove source_order block from the source order linked list:
        remove_block(*block_pointer, source_order);
        // Insert new lower order blocks to the lower order linked list:
        insert_block(new_LHS, source_order - 1);
        insert_block(new_RHS, source_order - 1);
        return new_LHS;
    }

    /**
     * Takes a block in the given source order, and merges it (and its buddy) into the next order.
     * @param block_pointer A pointer to a pointer containing a block in the pair to merge.
     * @param source_order The order in which the pair of blocks live.
     * @return Returns the new slot that points to the merged block.
     */
    PageDescriptor **merge_block(PageDescriptor **block_pointer, int source_order)
    {

        // check alignment of pgd in source_order:
        if (!is_aligned(*block_pointer, source_order)) {
            syslog.message(LogLevel::ERROR, "Page descriptor is not aligned");
            return NULL;
        }
        if (source_order == MAX_ORDER) {
            syslog.message(LogLevel::INFO, "Cannot merge blocks of max order");
            return block_pointer;
        }

        PageDescriptor* source_order_buddy = buddy_of(*block_pointer, source_order);
        remove_block(*block_pointer, source_order);
        remove_block(source_order_buddy, source_order);
        PageDescriptor* new_block = (*block_pointer < source_order_buddy) ? *block_pointer : source_order_buddy;
        return insert_block(new_block, source_order + 1);
    }

    /**
     * Insert a node into sorted linked list 
     * @param head head of linked list
     * @param node node to be inserted
     */
    void insert(PageDescriptor **head, PageDescriptor *node)
    {
        PageDescriptor **curr;
        for (curr = head; *curr; curr = &(*curr)->next_free)
        {
            if (*curr >= node)
            {
                break;
            }
        }
        node->next_free = *curr;
        *curr = node;
    }

    /**
     * Remove a node from linked list
     * @param head head of linked list
     * @param node node to be removed
     * @return true if node is found in linked list
     * @return false if node is not found in linked list
     */
    bool remove(PageDescriptor **head, PageDescriptor *node)
    {
        PageDescriptor **curr;
        for (curr = head; *curr; curr = &(*curr)->next_free)
        {
            if (*curr == node)
            {
                *curr = (*curr)->next_free;
                return true;
            }
        }
        return false;
    }

public:
    /**
     * Allocates 2^order number of contiguous pages
     * @param order The power of two, of the number of contiguous pages to allocate.
     * @return Returns a pointer to the first page descriptor for the newly allocated page range, or NULL if
     * allocation failed.
     */
    PageDescriptor *allocate_pages(int order) override
    {
        // find from order to MAX_ORDER
        for (int source_order = order; source_order <= MAX_ORDER; ++source_order)
        {
            // make sure it isnt null
            if (_free_areas[source_order])
            {
                PageDescriptor *pages = _free_areas[source_order];
                // remove pages by changing pointer
                _free_areas[source_order] = pages->next_free;
                // split pages until required order
                while (source_order != order)
                {
                    // insert buddy pages
                    source_order--;
                    insert(&_free_areas[source_order], buddy_of(pages, source_order));
                }
                return pages;
            }
        }
        // null if alloc failed
        return nullptr;
    }

    /**
     * Frees 2^order contiguous pages.
     * @param pgd A pointer to an array of page descriptors to be freed.
     * @param order The power of two number of contiguous pages to free.
     */
    void free_pages(PageDescriptor *pgd, int order) override
    {
        // merge pages with its buddy
        while (order < MAX_ORDER)
        {
            PageDescriptor *buddy = buddy_of(pgd, order);
            // try remove buddy
            if (!remove(&_free_areas[order], buddy))
            {
                break;
            }
            // merge into higher order
            ++order;
            // update leftmost pages after merged
            pgd = MIN(pgd, buddy);
        }
        // no further merge possible, so just insert pages
        insert(&_free_areas[order], pgd);
    }

    /**
     * Marks a range of pages as available for allocation.
     * @param start A pointer to the first page descriptors to be made available.
     * @param count The number of page descriptors to make available.

    */

    virtual void insert_page_range(PageDescriptor *start, uint64_t count) override
    {

        PageDescriptor* pgd_ptr = start;
        uint64_t pages_left= count;
        while (pages_left> 0) {
            int order = MAX_ORDER;
            // decrement order until pgd_ptr aligns with order:
            while (!is_aligned(pgd_ptr, order) and order >= 0) {
                order --;
            }
            while (get_block_size(order) > pages_left and order >= 0) {
                order --;
            }
            // mark block as available for allocation:
            uint64_t block_size = get_block_size(order);
            insert_block(pgd_ptr, order);
            pgd_ptr += block_size;  // move to next block
            pages_left-= block_size;
        }
    }




    /**
     * Marks a range of pages as unavailable for allocation.
     * @param start A pointer to the first page descriptors to be made unavailable.
     * @param count The number of page descriptors to make unavailable.
     */
    virtual void remove_page_range(PageDescriptor *start, uint64_t count) override
    {
        // do nothing if count == 0
        if (count == 0)
        {
            return;
        }
        
        PageDescriptor *end = start + count;
        // iterate by order of _free_areas
        for (int order = 0; order <= MAX_ORDER; ++order)
        {
            // iterate thru linked list of blocks of given order 
            for (PageDescriptor **curr = &_free_areas[order]; *curr;)
            {
                PageDescriptor *block = *curr, *block_end = block +  get_block_size(order);
                // check for overlap
                if (block < end && start < block_end)
                {
                    // remove block
                    *curr = block->next_free;
                    // insert [block, start)
                    if (block < start)
                    {
                        insert_page_range(block, start - block);
                    }
                    // insert [end, block_end)
                    if (end < block_end)
                    {
                        insert_page_range(end, block_end - end);
                    }
                }
                //move to next
                else
                {
                    curr = &block->next_free;
                }
            }
        }
    }

    /**
     * Initialises the allocation algorithm.
     * @return Returns TRUE if the algorithm was successfully initialised, FALSE otherwise.
     */
    bool init(PageDescriptor *page_descriptors, uint64_t nr_page_descriptors) override
    {
        _pgd_base = page_descriptors;
        _nr_pgds = nr_page_descriptors;
        _pgd_last = _pgd_base + _nr_pgds;
        return true;
    }

    /**
     * Returns the friendly name of the allocation algorithm, for debugging and selection purposes.
     */
    const char *name() const override { return "buddy"; }

    /**
     * Dumps out the current state of the buddy system
     */
    void dump_state() const override
    {
        // Print out a header, so we can find the output in the logs.
        mm_log.messagef(LogLevel::DEBUG, "BUDDY STATE:");

        // Iterate over each free area.
        for (unsigned int i = 0; i < ARRAY_SIZE(_free_areas); i++)
        {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "[%d] ", i);

            // Iterate over each block in the free area.
            PageDescriptor *pg = _free_areas[i];
            while (pg)
            {
                // Append the PFN of the free block to the output buffer.
                snprintf(buffer, sizeof(buffer), "%s%lx ", buffer, sys.mm().pgalloc().pgd_to_pfn(pg));
                pg = pg->next_free;
            }

            mm_log.messagef(LogLevel::DEBUG, "%s", buffer);
        }
    }

private:
    PageDescriptor *_free_areas[MAX_ORDER + 1];
    PageDescriptor *_pgd_base;
    PageDescriptor* _pgd_last;
    uint64_t _nr_pgds;
};

/* --- DO NOT CHANGE ANYTHING BELOW THIS LINE --- */

/*
 * Allocation algorithm registration framework
 */
RegisterPageAllocator(BuddyPageAllocator);