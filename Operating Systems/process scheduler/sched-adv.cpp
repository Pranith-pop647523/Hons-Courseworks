/*
 * The Priority Task Scheduler
 * SKELETON IMPLEMENTATION TO BE FILLED IN FOR TASK 1
 */

#include <infos/kernel/sched.h>
#include <infos/kernel/thread.h>
#include <infos/kernel/log.h>
#include <infos/util/list.h>
#include <infos/util/map.h>
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
    const char* name() const override { return "adv"; }

    /**
     * Called during scheduler initialisation.
     */
    void init()
    {
        //idk
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

            HighQ.enqueue(&entity);
    
          }

          else if (priority == INTERACTIVE ){
            HighQ.enqueue(&entity) ;
          }

         else if (priority ==NORMAL ){
            LowQ.enqueue(&entity);
          }
         else if (priority ==  DAEMON ){
            LowQ.enqueue(&entity);
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

            HighQ.remove(&entity) ;
          }
          else if (priority == INTERACTIVE ){
            HighQ.remove(&entity) ;
          }
         else if (priority == NORMAL ){
            LowQ.remove(&entity);
          }
         else if (priority == DAEMON ){
            LowQ.remove(&entity);
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


        turns = (turns+1)%9; //turn counter resets after 9 


        

        if (HighQ.count() + LowQ.count() == 0) { //return null if no processes
            return NULL;
        }

        else if (HighQ.empty() and !LowQ.empty() ){ //return highq element if lowq empty
           // syslog.messagef(LogLevel::DEBUG, "%d",1);
            return RRSelection(LowQ);
        }
        else if (LowQ.empty() and !HighQ.empty() ){ //return highq element if lowq empty
            //syslog.messagef(LogLevel::DEBUG, "%d",2);
            return RRSelection(HighQ);
        }

        else if (!HighQ.empty() and turns < 7){ //return highq element if within its turn

            //syslog.messagef(LogLevel::DEBUG, "%d",3);

            return RRSelection(HighQ);
        }
        else if (!LowQ.empty() and turns >= 7){ //return lowq element if within its turn

           // syslog.messagef(LogLevel::DEBUG, "%d",4);
            return RRSelection(LowQ);
        }
        else{
            return NULL;
        }
        







    }


private:
        List<SchedulingEntity *>  HighQ;
        List<SchedulingEntity *> LowQ;
        int turns = -1;

  
        
        
};

/* --- DO NOT CHANGE ANYTHING BELOW THIS LINE --- */

RegisterScheduler(MultipleQueuePriorityScheduler);