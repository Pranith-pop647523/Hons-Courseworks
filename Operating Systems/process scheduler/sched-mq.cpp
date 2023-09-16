/*
 * The Priority Task Scheduler
 * SKELETON IMPLEMENTATION TO BE FILLED IN FOR TASK 1
 */

#include <infos/kernel/sched.h>
#include <infos/kernel/thread.h>
#include <infos/kernel/log.h>
#include <infos/util/list.h>
#include <infos/util/lock.h>


using namespace infos::kernel;
using namespace infos::util;



/**
 * A Multiple Queue priority scheduling algorithm
 */
class MultipleQueuePriorityScheduler : public SchedulingAlgorithm
{
    
public:

    /**
     * Returns the friendly name of the algorithm, for debugging and selection purposes.
     */
    const char* name() const override { return "mq"; }

    /**
     * Called during scheduler initialisation.
     */
    void init()
    {
       
        // zzzzzzz
    }

    /**
     * Called when a scheduling entity becomes eligible for running.
     * @param entity
     */
    void add_to_runqueue(SchedulingEntity& entity) override
    {
        UniqueIRQLock L;
        SchedulingEntityPriority::SchedulingEntityPriority priority = entity.priority();
        using namespace SchedulingEntityPriority;
        if (priority == REALTIME ){

            realTimeQ.enqueue(&entity) ;
          }
          else if (priority == INTERACTIVE ){
            interactiveQ.enqueue(&entity) ;
          }

         else if (priority ==NORMAL ){
            normalQ.enqueue(&entity);
          }
         else if (priority ==  DAEMON ){
            daemonQ.enqueue(&entity);
          }
        else {
            syslog.messagef(LogLevel::ERROR, "Priority does not exist");
        }

    }

    /**
     * Called when a scheduling entity is no longer eligible for running.
     * @param entity
     */
    void remove_from_runqueue(SchedulingEntity& entity) override

    {
        UniqueIRQLock l;
        SchedulingEntityPriority::SchedulingEntityPriority priority = entity.priority();
        using namespace SchedulingEntityPriority;
        if (priority == REALTIME ){

            realTimeQ.remove(&entity) ;
          }
          else if (priority == INTERACTIVE ){
            interactiveQ.remove(&entity) ;
          }
         else if (priority == NORMAL ){
            normalQ.remove(&entity);
          }
         else if (priority == DAEMON ){
            daemonQ.remove(&entity);
          }
        else {
            syslog.messagef(LogLevel::ERROR, "Priority does not exist");
        }
    }



    /**
     * Executes a process for a given time slice and sends it to the back of 
     * the queue if it doesnt finish within the time slice.
     * @param processQ
     */


    SchedulingEntity* RRSelection(List<SchedulingEntity *> &processQ) {
        if (processQ.count() == 1) {
            return processQ.first();
        }

        UniqueIRQLock l;// playing with queue elements so we need a lock

        SchedulingEntity* entity = processQ.dequeue();
        processQ.enqueue(entity);
        return entity;
    }

    /**
     * Called every time a scheduling event occurs, to cause the next eligible entity
     * to be chosen.  The next eligible entity might actually be the same entity, if
     * e.g. its timeslice has not expired.
     */

    

    SchedulingEntity *pick_next_entity() override
    
    {

    using namespace infos::kernel::SchedulingEntityPriority ;
      
      
      // picks entity based on priorities. picks a lower priority entity only if all higher priorities are empty

            


        if (!realTimeQ.empty()){
          return RRSelection(realTimeQ) ;
        }
        else if (!interactiveQ.empty()){
           return RRSelection(interactiveQ) ;
        }
        else if  (!normalQ.empty()){
          return RRSelection(normalQ) ;
        }
        else if  (!daemonQ.empty()){
          return RRSelection(daemonQ) ;
        }
        else{
            return NULL;
        }


        }
        
      
    


private:
        List<SchedulingEntity *>  realTimeQ;
        List<SchedulingEntity *> interactiveQ;
        List<SchedulingEntity *> normalQ;
        List<SchedulingEntity *>  daemonQ;

};

/* --- DO NOT CHANGE ANYTHING BELOW THIS LINE --- */

RegisterScheduler(MultipleQueuePriorityScheduler);