function Settings({ register }) {
    return (
        <div className="window font-sans mt-3 text-black dark:text-white row-span-1">
            <h2 className="text-2xl md:text-3xl text-black dark:text-white font-bold mt-1">Settings</h2>
            <div class="grid gap-6 mb-6 lg:grid-cols-2">
                <div>
                    <label for="num_beams" class="block mb-2 text-sm font-medium text-gray-900 dark:text-gray-300">num_beams</label>
                    <input {...register("num_beams")} type="number" id="num_beams" class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="integer" required defaultValue={3}></input>
                </div>
                <div>
                    <label for="repetition_penalty" class="block mb-2 text-sm font-medium text-gray-900 dark:text-gray-300">repetition_penalty</label>
                    <input {...register("repetition_penalty")} type="number" step={0.01} id="repitition_penalty" class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="integer" required defaultValue={2.75}></input>
                </div>
                <div>
                    <label for="top_p" class="block mb-2 text-sm font-medium text-gray-900 dark:text-gray-300">top_p</label>
                    <input {...register("top_p")} type="number" id="repitition_penalty" class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="integer" required defaultValue={50}></input>
                </div>
                <div>
                    <label for="top_k" class="block mb-2 text-sm font-medium text-gray-900 dark:text-gray-300">top_k</label>
                    <input {...register("top_k")} type="number" id="repitition_penalty" class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="integer" required defaultValue={20}></input>
                </div>
            </div>
        </div>
    )
}

export default Settings
