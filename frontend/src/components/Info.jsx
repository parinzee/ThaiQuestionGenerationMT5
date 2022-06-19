function Info() {
    return (
        <div className="shadow-xl shadow-blue-500/50 hover:shadow-indigo-500/50 window font-sans m-8 text-black dark:text-white row-span-1 overflow-auto max-h-max">
            <h1 className="text-3xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-500">Automatic Thai Question Generation with MT5</h1>
            <h2 className="text-lg md:text-xl text-gray-700 dark:text-gray-400 mt-1">Created by: Parinthapat Pengpun</h2>
            <p className="text-lg md:text-xl dark:text-white text-black mt-5">This is an <b>mT5-small</b> model that has been finetuned to <b>generate questions</b> utilizing <b>only the context.</b> More info: GitHub, Blog.</p>
            <hr className="mt-5" />
            <h2 className="text-2xl md:text-3xl text-black dark:text-white font-bold mt-1">Variants</h2>
            <ul className="text-lg md:text-xl mt-1 list-disc list-inside">
                <li><span className="font-mono">aug:</span> Models with the dataset augmented (modified for better functionality)</li>
                <li><span className="font-mono">sep:</span> Generate questions with &quot;&lt;sep&gt;&quot; instead of &quot;1.&quot; for separation</li>
                <li><span className="font-mono">numsep:</span> Generate questions with &quot;&lt;1&gt;&quot; instead of &quot;1.&quot; for separation</li>
            </ul>
            <hr className="mt-5" />
            <h2 className="text-2xl md:text-3xl text-black dark:text-white font-bold mt-1">Metrics</h2>
            <table className="mt-3 table-fixed border-spacing-3 w-full border-separate border border-slate-500 overflow-scroll">
                <thead>
                    <tr className="text-xl">
                        <th>Model</th>
                        <th>Meteor</th>
                        <th>GLEU</th>
                        <th>BLEU</th>
                        <th>CHRF</th>
                        <th>ROUGE-L</th>
                    </tr>
                </thead>
                <tbody className="text-center">
                    <tr>
                        <td>Baseline</td>
                        <td>0.46</td>
                        <td>0.20</td>
                        <td>0.16</td>
                        <td>0.32</td>
                        <td>0.57</td>
                    </tr>
                    <tr>
                        <td>Default</td>
                        <td>0.56</td>
                        <td>0.34</td>
                        <td>0.31</td>
                        <td>0.46</td>
                        <td>0.86</td>
                    </tr>
                    <tr>
                        <td>numsep</td>
                        <td>0.59</td>
                        <td>0.36</td>
                        <td>0.34</td>
                        <td>0.46</td>
                        <td>0.81</td>
                    </tr>
                </tbody>
            </table>
        </div>
    )
}

export default Info