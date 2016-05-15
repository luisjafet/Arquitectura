/*
 * cache.c
 */


#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "cache.h"
#include "main.h"

/* cache configuration parameters */
static int cache_split = 0;
static int cache_usize = DEFAULT_CACHE_SIZE;
static int cache_isize = DEFAULT_CACHE_SIZE; 
static int cache_dsize = DEFAULT_CACHE_SIZE;
static int cache_block_size = DEFAULT_CACHE_BLOCK_SIZE;
static int words_per_block = DEFAULT_CACHE_BLOCK_SIZE / WORD_SIZE;
static int cache_assoc = DEFAULT_CACHE_ASSOC;
static int cache_writeback = DEFAULT_CACHE_WRITEBACK;
static int cache_writealloc = DEFAULT_CACHE_WRITEALLOC;

/* cache model data structures */
static Pcache icache;
static Pcache dcache;
static cache c1;
static cache c2;
static cache_stat cache_stat_inst;
static cache_stat cache_stat_data;

/************************************************************/
void set_cache_param(param, value)
  int param;
  int value;
{

  switch (param) {
  case CACHE_PARAM_BLOCK_SIZE:
    cache_block_size = value;
    words_per_block = value / WORD_SIZE;
    break;
  case CACHE_PARAM_USIZE:
    cache_split = FALSE;
    cache_usize = value;
    break;
  case CACHE_PARAM_ISIZE:
    cache_split = TRUE;
    cache_isize = value;
    break;
  case CACHE_PARAM_DSIZE:
    cache_split = TRUE;
    cache_dsize = value;
    break;
  case CACHE_PARAM_ASSOC:
    cache_assoc = value;
    break;
  case CACHE_PARAM_WRITEBACK:
    cache_writeback = TRUE;
    break;
  case CACHE_PARAM_WRITETHROUGH:
    cache_writeback = FALSE;
    break;
  case CACHE_PARAM_WRITEALLOC:
    cache_writealloc = TRUE;
    break;
  case CACHE_PARAM_NOWRITEALLOC:
    cache_writealloc = FALSE;
    break;
  default:
    printf("error set_cache_param: bad parameter value\n");
    exit(-1);
  }

}
/************************************************************/

/************************************************************/
void init_cache(){
	unsigned aux_mask1, aux_mask2;
	if(cache_split == 0){
		// 1 cache
		c1.size = cache_usize/words_per_block;	
		c1.n_sets = cache_usize/cache_block_size/cache_assoc;
	}else{
		// inst cache
		c1.size = cache_isize/words_per_block;
		c1.n_sets = cache_isize/cache_block_size/cache_assoc;
		
		// data cache
		c2.size = cache_dsize/words_per_block;
		c2.n_sets = cache_dsize/cache_block_size/cache_assoc;
		aux_mask2 = c2.n_sets-1;
	}
	aux_mask1 = c1.n_sets-1;

	// we need this for cache 1
	c1.index_mask = (aux_mask1) << LOG2(cache_block_size);
	c1.index_mask_offset = LOG2(cache_block_size);	
	c1.LRU_head = (Pcache_line *)malloc(sizeof(Pcache_line)*c1.n_sets);
	c1.LRU_tail = (Pcache_line *)malloc(sizeof(Pcache_line)*c1.n_sets);
	c1.set_contents = (int *)malloc(sizeof(int)*c1.n_sets);
	int i;
	for(i = 0; i < c1.n_sets; i++){
		c1.LRU_head[i] = NULL;
		c1.LRU_tail[i] = NULL;
		c1.set_contents[i] = 0;
	}

	// for cache 2
	if(cache_split != 0){
		c2.index_mask = (aux_mask2) << LOG2(cache_block_size);
		c2.index_mask_offset = LOG2(cache_block_size);	
		c2.LRU_head = (Pcache_line *)malloc(sizeof(Pcache_line)*c2.n_sets);
		c2.LRU_tail = (Pcache_line *)malloc(sizeof(Pcache_line)*c2.n_sets);
		c2.set_contents = (int *)malloc(sizeof(int)*c2.n_sets);
		int i;
		for(i = 0; i < c2.n_sets; i++){
			c2.LRU_head[i] = NULL;
			c2.LRU_tail[i] = NULL;
			c2.set_contents[i] = 0;
		}
	}

	// stats for data cache
	cache_stat_data.accesses = 0;
	cache_stat_data.misses = 0;
	cache_stat_data.replacements = 0;
	cache_stat_data.demand_fetches = 0;
	cache_stat_data.copies_back = 0;

	// stats for inst cache
	cache_stat_inst.accesses = 0;
	cache_stat_inst.misses = 0;
	cache_stat_inst.replacements = 0;
	cache_stat_inst.demand_fetches = 0;
	cache_stat_inst.copies_back = 0;
}
/************************************************************/

/************************************************************/
void perform_access(addr, access_type)
  unsigned addr, access_type;
{
	unsigned int index;
	unsigned tag;

	// We need cache 1 if inst trace or just 1 cache
	if(!cache_split || access_type == TRACE_INST_LOAD){
		// we need this for cache 1
		index = (addr & c1.index_mask) >> c1.index_mask_offset;
		tag = addr >> (LOG2(c1.n_sets) + LOG2(cache_block_size));
		// cache is not empty in this set
		if(c1.LRU_head[index] != NULL){
			Pcache_line current_line;
			// we have to check in the list
			for(current_line = c1.LRU_head[index]; current_line != c1.LRU_tail[index]->LRU_next; current_line = current_line->LRU_next){
				// cache hit! so we exit exec with return
				if(current_line->tag == tag){
			        if(access_type == TRACE_DATA_LOAD || access_type == TRACE_DATA_STORE){
			        	cache_stat_data.accesses++;
			            if(access_type == TRACE_DATA_STORE){
							if(cache_writeback == TRUE){
				            	current_line->dirty = TRUE;							
							}else{
								cache_stat_data.copies_back++;
							}
			            }
			        }
			        else if(access_type == TRACE_INST_LOAD){
			            cache_stat_inst.accesses++;
			        }
					Pcache_line line = current_line;
					delete(&c1.LRU_head[index], &c1.LRU_tail[index], current_line);
					insert(&c1.LRU_head[index], &c1.LRU_tail[index], line);
					return;
				}
			}
			// cache miss we need to add element to cache
			Pcache_line line = malloc(sizeof(cache_line));
			// we can append element to cache line list
			if(c1.set_contents[index] < cache_assoc){
				line->tag = tag;
				if(access_type == TRACE_DATA_LOAD || access_type == TRACE_DATA_STORE){
					cache_stat_data.accesses++;
					cache_stat_data.misses++;
					if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
						cache_stat_data.demand_fetches += words_per_block;									
					}
					if(access_type == TRACE_DATA_STORE && cache_writealloc == TRUE){
						if(cache_writeback == TRUE){
							line->dirty = TRUE;						
						}else{
							cache_stat_data.copies_back++;
						}
					}
					else if(access_type == TRACE_DATA_STORE && cache_writealloc == FALSE){
						cache_stat_data.copies_back++;
					}
				}else if(access_type == TRACE_INST_LOAD){
					cache_stat_inst.accesses++;
					cache_stat_inst.misses++;
					if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
						cache_stat_inst.demand_fetches += words_per_block;
					}				
				}
				if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
					// we insert and increase set contents
					insert(&c1.LRU_head[index], &c1.LRU_tail[index], line);	
					c1.set_contents[index]++;	
				}
			}else{
				// we can not append element to cache line list, we need to replace
				line->dirty = FALSE;
				line->tag = tag;
				if(access_type == TRACE_DATA_LOAD || access_type == TRACE_DATA_STORE){
					cache_stat_data.accesses++;
					cache_stat_data.misses++;
					if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
						cache_stat_data.replacements++;
						cache_stat_data.demand_fetches += words_per_block;	
					}	
					if(access_type == TRACE_DATA_STORE && cache_writealloc == TRUE){
						if(cache_writeback == TRUE) {
							line->dirty = TRUE;						
						}else{
							cache_stat_data.copies_back++;
						}
					}
					else if(access_type == TRACE_DATA_STORE && cache_writealloc == FALSE){
						cache_stat_data.copies_back++;					
					}
				}else if(access_type == TRACE_INST_LOAD){
					cache_stat_inst.accesses++;
					cache_stat_inst.misses++;
					if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
						cache_stat_inst.replacements++;
						cache_stat_inst.demand_fetches += words_per_block;	
					}				
				}
				if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
					if(c1.LRU_tail[index]->dirty == TRUE && (access_type == TRACE_DATA_LOAD || access_type == TRACE_DATA_STORE)){
						cache_stat_data.copies_back += words_per_block;
					}
					else if(c1.LRU_tail[index]->dirty == TRUE && access_type == TRACE_INST_LOAD){
						cache_stat_inst.copies_back += words_per_block;
					}
					delete(&c1.LRU_head[index], &c1.LRU_tail[index], c1.LRU_tail[index]);
					insert(&c1.LRU_head[index], &c1.LRU_tail[index], line);
				}
			}
		}else{
			// cache is empty in this set, it is also a cache miss
			Pcache_line line = malloc(sizeof(cache_line));
			line->tag = tag;
			if(access_type == TRACE_DATA_LOAD || access_type == TRACE_DATA_STORE){
				cache_stat_data.accesses++;
				cache_stat_data.misses++;
				if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
					cache_stat_data.demand_fetches += words_per_block;				
				}
				if(access_type == TRACE_DATA_STORE && cache_writealloc == TRUE){
					if(cache_writeback == TRUE){
						line->dirty = TRUE;					
					}else{
						cache_stat_data.copies_back++;
					}
				}else if(access_type == TRACE_DATA_STORE && cache_writealloc == FALSE){
					cache_stat_data.copies_back++;				
				}
			}else if(access_type == TRACE_INST_LOAD){
				cache_stat_inst.accesses++;
				cache_stat_inst.misses++;
				if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
					cache_stat_inst.demand_fetches += words_per_block;				
				}
			}
			if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
				// we insert and increase set contents
				insert(&c1.LRU_head[index], &c1.LRU_tail[index], line);			
				c1.set_contents[index]++;
			}
		}
	// for cache 2
	}else{
		index = (addr & c2.index_mask) >> c2.index_mask_offset;
		tag = addr >> (LOG2(c2.n_sets) + LOG2(cache_block_size));
		// cache is not empty in this set
		if(c2.LRU_head[index] != NULL){
			Pcache_line current_line;
			// we have to check in the list
			for(current_line = c2.LRU_head[index]; current_line != c2.LRU_tail[index]->LRU_next; current_line = current_line->LRU_next){
				// cache hit! so we exit exec with return
				if(current_line->tag == tag){
			        if(access_type == TRACE_DATA_LOAD || access_type == TRACE_DATA_STORE){
			        	cache_stat_data.accesses++;
			            if(access_type == TRACE_DATA_STORE){
							if(cache_writeback == TRUE){
				            	current_line->dirty = TRUE;							
							}else{
								cache_stat_data.copies_back++;
							}
			            }
			        }
			        else if(access_type == TRACE_INST_LOAD){
			            cache_stat_inst.accesses++;
			        }
					Pcache_line line = current_line;
					delete(&c2.LRU_head[index], &c2.LRU_tail[index], current_line);
					insert(&c2.LRU_head[index], &c2.LRU_tail[index], line);
					return;
				}
			}
			// cache miss we need to add element to cache
			Pcache_line line = malloc(sizeof(cache_line));
			// we can append element to cache line list
			if(c2.set_contents[index] < cache_assoc){
				line->tag = tag;
				if(access_type == TRACE_DATA_LOAD || access_type == TRACE_DATA_STORE){
					cache_stat_data.accesses++;
					cache_stat_data.misses++;
					if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
						cache_stat_data.demand_fetches += words_per_block;									
					}
					if(access_type == TRACE_DATA_STORE && cache_writealloc == TRUE){
						if(cache_writeback == TRUE){
							line->dirty = TRUE;						
						}else{
							cache_stat_data.copies_back++;
						}
					}
					else if(access_type == TRACE_DATA_STORE && cache_writealloc == FALSE){
						cache_stat_data.copies_back++;
					}
				}else if(access_type == TRACE_INST_LOAD){
					cache_stat_inst.accesses++;
					cache_stat_inst.misses++;
					if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
						cache_stat_inst.demand_fetches += words_per_block;
					}				
				}
				if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
					// we insert and increase set contents
					insert(&c2.LRU_head[index], &c2.LRU_tail[index], line);	
					c2.set_contents[index]++;	
				}
			}else{
				// we can not append element to cache line list, we need to replace
				line->dirty = FALSE;
				line->tag = tag;
				if(access_type == TRACE_DATA_LOAD || access_type == TRACE_DATA_STORE){
					cache_stat_data.accesses++;
					cache_stat_data.misses++;
					if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
						cache_stat_data.replacements++;
						cache_stat_data.demand_fetches += words_per_block;	
					}	
					if(access_type == TRACE_DATA_STORE && cache_writealloc == TRUE){
						if(cache_writeback == TRUE) {
							line->dirty = TRUE;						
						}else{
							cache_stat_data.copies_back++;
						}
					}
					else if(access_type == TRACE_DATA_STORE && cache_writealloc == FALSE){
						cache_stat_data.copies_back++;					
					}
				}else if(access_type == TRACE_INST_LOAD){
					cache_stat_inst.accesses++;
					cache_stat_inst.misses++;
					if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
						cache_stat_inst.replacements++;
						cache_stat_inst.demand_fetches += words_per_block;	
					}				
				}
				if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
					if(c2.LRU_tail[index]->dirty == TRUE && (access_type == TRACE_DATA_LOAD || access_type == TRACE_DATA_STORE)){
						cache_stat_data.copies_back += words_per_block;
					}
					else if(c2.LRU_tail[index]->dirty == TRUE && access_type == TRACE_INST_LOAD){
						cache_stat_inst.copies_back += words_per_block;
					}
					delete(&c2.LRU_head[index], &c2.LRU_tail[index], c2.LRU_tail[index]);
					insert(&c2.LRU_head[index], &c2.LRU_tail[index], line);
				}
			}
		}else{
			// cache is empty in this set, it is also a cache miss
			Pcache_line line = malloc(sizeof(cache_line));
			line->tag = tag;
			if(access_type == TRACE_DATA_LOAD || access_type == TRACE_DATA_STORE){
				cache_stat_data.accesses++;
				cache_stat_data.misses++;
				if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
					cache_stat_data.demand_fetches += words_per_block;				
				}
				if(access_type == TRACE_DATA_STORE && cache_writealloc == TRUE){
					if(cache_writeback == TRUE){
						line->dirty = TRUE;					
					}else{
						cache_stat_data.copies_back++;
					}
				}else if(access_type == TRACE_DATA_STORE && cache_writealloc == FALSE){
					cache_stat_data.copies_back++;				
				}
			}else if(access_type == TRACE_INST_LOAD){
				cache_stat_inst.accesses++;
				cache_stat_inst.misses++;
				if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
					cache_stat_inst.demand_fetches += words_per_block;				
				}
			}
			if(cache_writealloc == TRUE || access_type != TRACE_DATA_STORE){
				// we insert and increase set contents
				insert(&c2.LRU_head[index], &c2.LRU_tail[index], line);			
				c2.set_contents[index]++;
			}
		}
	}
}
/************************************************************/

/************************************************************/
void flush()
{	
	int i;
	// we need this for 1 or 2 caches
	for(i = 0; i < c1.n_sets; i++){
		Pcache_line current_line;
		if(c1.LRU_head[i] != NULL){
			for(current_line = c1.LRU_head[i]; current_line != c1.LRU_tail[i]->LRU_next; current_line = current_line->LRU_next){
				if(current_line != NULL && current_line->dirty == TRUE){
					cache_stat_inst.copies_back += words_per_block;
				}
			}
		}
	}

	// we only need this for cache 2
	if(cache_split != 0){
		for(i = 0; i < c2.n_sets; i++){
			Pcache_line current_line;
			if(c2.LRU_head[i] != NULL){
				for(current_line = c2.LRU_head[i]; current_line != c2.LRU_tail[i]->LRU_next; current_line = current_line->LRU_next){
					if(current_line != NULL && current_line->dirty == TRUE){
						cache_stat_inst.copies_back += words_per_block;
					}
				}
			}
		}
	}
}
/************************************************************/

/************************************************************/
void delete(head, tail, item)
  Pcache_line *head, *tail;
  Pcache_line item;
{
  if (item->LRU_prev) {
    item->LRU_prev->LRU_next = item->LRU_next;
  } else {
    /* item at head */
    *head = item->LRU_next;
  }

  if (item->LRU_next) {
    item->LRU_next->LRU_prev = item->LRU_prev;
  } else {
    /* item at tail */
    *tail = item->LRU_prev;
  }
}
/************************************************************/

/************************************************************/
/* inserts at the head of the list */
void insert(head, tail, item)
  Pcache_line *head, *tail;
  Pcache_line item;
{
  item->LRU_next = *head;
  item->LRU_prev = (Pcache_line)NULL;

  if (item->LRU_next)
    item->LRU_next->LRU_prev = item;
  else
    *tail = item;

  *head = item;
}
/************************************************************/

/************************************************************/
void dump_settings()
{
  printf("Cache Settings:\n");
  if (cache_split) {
    printf("\tSplit I- D-cache\n");
    printf("\tI-cache size: \t%d\n", cache_isize);
    printf("\tD-cache size: \t%d\n", cache_dsize);
  } else {
    printf("\tUnified I- D-cache\n");
    printf("\tSize: \t%d\n", cache_usize);
  }
  printf("\tAssociativity: \t%d\n", cache_assoc);
  printf("\tBlock size: \t%d\n", cache_block_size);
  printf("\tWrite policy: \t%s\n", 
	 cache_writeback ? "WRITE BACK" : "WRITE THROUGH");
  printf("\tAllocation policy: \t%s\n",
	 cache_writealloc ? "WRITE ALLOCATE" : "WRITE NO ALLOCATE");
}
/************************************************************/

/************************************************************/
void print_stats()
{
  printf("*** CACHE STATISTICS ***\n");
  printf("  INSTRUCTIONS\n");
  printf("  accesses:  %d\n", cache_stat_inst.accesses);
  printf("  misses:    %d\n", cache_stat_inst.misses);
  printf("  miss rate: %f\n", 
	 (float)cache_stat_inst.misses / (float)cache_stat_inst.accesses);
  printf("  replace:   %d\n", cache_stat_inst.replacements);

  printf("  DATA\n");
  printf("  accesses:  %d\n", cache_stat_data.accesses);
  printf("  misses:    %d\n", cache_stat_data.misses);
  printf("  miss rate: %f\n", 
	 (float)cache_stat_data.misses / (float)cache_stat_data.accesses);
  printf("  replace:   %d\n", cache_stat_data.replacements);

  printf("  TRAFFIC (in words)\n");
  printf("  demand fetch:  %d\n", cache_stat_inst.demand_fetches + 
	 cache_stat_data.demand_fetches);
  printf("  copies back:   %d\n", cache_stat_inst.copies_back +
	 cache_stat_data.copies_back);
}
/************************************************************/
void print_cache() {
	int i;
	Pcache_line line;
	for(i = 0; i < c1.n_sets; i++) {
		for(line = c1.LRU_head[i]; line != c1.LRU_tail[i]; line = line->LRU_next) {
			printf("%d ", line->tag);
		}
		printf("\n");
	}
	printf("num_sets = %d\n", i);
}
/************************************************************/
void print_set_contents() {
	int i;
	for(i = 0; i < c1.n_sets; i++) {
		printf("%d\n", c1.set_contents[i]);
	}
}
/************************************************************/