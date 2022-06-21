import { motion } from "framer-motion";
function Info() {
  return (
    <motion.div
      className="transition-shadow md:h-full shadow-xl shadow-blue-500/50 hover:shadow-indigo-500/50 window font-sans text-black dark:text-white md:row-span-6 col-span-1 md:col-span-3 md:overflow-auto"
      initial="pageInitial"
      animate="pageAnimate"
      variants={{
        pageInitial: {
          opacity: 0,
        },
        pageAnimate: {
          opacity: 1,
          transition: {
            delay: 0.2,
            duration: 0.5,
          },
        },
      }}
    >
      <h1 className="text-3xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-500">
        Automatic Thai Question Generation with MT5
      </h1>
      <h2 className="text-lg md:text-xl text-gray-700 dark:text-gray-400 mt-1">
        Created by:{" "}
        <a
          target="_blank"
          href="https://github.com/parinzee"
          className="decoration-amber-600"
        >
          Parinthapat Pengpun
        </a>
      </h2>
      <p className="text-lg md:text-xl dark:text-white text-black mt-5">
        This is an <b>mT5-small</b> model that has been finetuned to{" "}
        <b>generate questions</b> utilizing <b>only the context.</b> More info:{" "}
        <a
          target="_blank"
          href="https://github.com/parinzee/ThaiQuestionGenerationMT5"
          className="decoration-teal-500"
        >
          GitHub
        </a>
        ,{" "}
        <a target="_blank" href="https://medium.com/@parinzee/studying-let-an-ai-generate-q-as-to-quiz-you-9ef27b1554d" className="decoration-indigo-500">
          Medium Blog
        </a>
        .
      </p>
      <hr className="mt-5" />
      <h2 className="text-2xl md:text-3xl text-black dark:text-white font-bold mt-1">
        Variants
      </h2>
      <ul className="text-lg md:text-xl mt-1 list-disc list-inside">
        <li>
          <span className="font-mono">Baseline:</span> This model was only
          trained with xquad
        </li>
        <li>
          <span className="font-mono">Default:</span> This model was trained on
          xquad, thaiqa, and iapp-wiki. Other models except baseline builds on
          this one.
        </li>
        <li>
          <span className="font-mono">aug:</span> Models with the dataset
          augmented (modified for better functionality)
        </li>
        <li>
          <span className="font-mono">numsep:</span> Generate questions with
          &quot;&lt;1&gt;&quot; instead of &quot;1.&quot; for separation
        </li>
      </ul>
      <hr className="mt-5" />
      <h2 className="text-2xl md:text-3xl text-black dark:text-white font-bold mt-1">
        Formatting
      </h2>
      <div className="text-lg md:text-xl dark:text-white text-black mt-5 font-mono p-2 bg-neutral-300 dark:bg-neutral-700 rounded-md max-w-max">
        สร้าง <i className="text-green-600 dark:text-lime-300 font-extrabold">จำนวน</i> คำถาม: <i className="text-green-600 dark:text-lime-300 font-extrabold">ข้อมูล</i>
      </div>
      <hr className="mt-5" />
      <h2 className="text-2xl md:text-3xl text-black dark:text-white font-bold mt-1">
        Metrics
      </h2>
      <div className="mt-2 overflow-auto">
        <table className="table-fixed border-spacing-3 w-[545px] md:w-full border-separate border border-slate-500">
          <thead>
            <tr className="text-xl">
              <th>Model</th>
              <th>Meteor</th>
              <th>GLEU</th>
              <th>BLEU-4</th>
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
            <tr>
              <td>aug-numsep</td>
              <td>0.60</td>
              <td>0.37</td>
              <td>0.36</td>
              <td>0.47</td>
              <td>0.84</td>
            </tr>
          </tbody>
        </table>
      </div>
    </motion.div>
  );
}

export default Info;
