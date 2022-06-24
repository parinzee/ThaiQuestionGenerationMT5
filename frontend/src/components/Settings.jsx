import { motion } from "framer-motion";

function Settings({ register }) {
  return (
    <motion.div className="transition-shadow window shadow-xl shadow-fuchsia-500/50 hover:shadow-blue-300/50 font-sans mt-3 text-black dark:text-white row-span-1"
      initial="pageInitial"
      animate="pageAnimate"
      variants={{
        pageInitial: {
          opacity: 0,
        },
        pageAnimate: {
          opacity: 1,
          transition: {
            delay: 0.6,
            duration: 0.5
          },
        },
      }}
    >
      <h1 className="text-2xl md:text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-sky-500 pb-2">
        Settings
      </h1>
      <div class="grid gap-4 lg:grid-cols-2">
        <div>
          <label
            for="num_beams"
            class="block mb-2 text-sm font-medium text-gray-900 dark:text-gray-300"
          >
            num_beams
          </label>
          <input
            {...register("num_beams")}
            type="number"
            id="num_beams"
            class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
            placeholder="integer"
            required
            defaultValue={4}
          ></input>
        </div>
        <div>
          <label
            for="repetition_penalty"
            class="block mb-2 text-sm font-medium text-gray-900 dark:text-gray-300"
          >
            repetition_penalty
          </label>
          <input
            {...register("repetition_penalty")}
            type="number"
            step={0.01}
            id="repitition_penalty"
            class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
            placeholder="integer"
            required
            defaultValue={3.1}
          ></input>
        </div>
        <div>
          <label
            for="top_p"
            class="block mb-2 text-sm font-medium text-gray-900 dark:text-gray-300"
          >
            top_p
          </label>
          <input
            {...register("top_p")}
            type="number"
            id="repitition_penalty"
            class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
            placeholder="integer"
            required
            defaultValue={50}
          ></input>
        </div>
        <div>
          <label
            for="top_k"
            class="block mb-2 text-sm font-medium text-gray-900 dark:text-gray-300"
          >
            top_k
          </label>
          <input
            {...register("top_k")}
            type="number"
            id="repitition_penalty"
            class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
            placeholder="integer"
            required
            defaultValue={20}
          ></input>
        </div>
      </div>
    </motion.div>
  );
}

export default Settings;
