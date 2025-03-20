# System Design Interview An Insider Guide
### Chapter 1 (SCALE FROM ZERO TO MILLIONS OF USERS)
Client, DNS, Web Servers, Cache, CDN, Databases (with sharding), Load Balancer, Message Queue, Logging, Metrics, Automation  
### Chapter 2 (BACK-OF-THE-ENVELOPE ESTIMATION)
Always compress data before sending over internet (Compression is fast), QPS, Peak QPS, Availability
### Chapter 3 (A FRAMEWORK FOR SYSTEM DESIGN INTERVIEWS)
Understand requirements before starting to answer -> Give High Level Design (Avoid Overengineering) -> Progress into Deep Dive -> Usually may ask you what is your design bottlenecks, server failures, even more scaling
### Chapter 4 (DESIGN A RATE LIMITER)
Making the rate limiter as middleware between client and web servers. Rate limiting information will be stored in in-memory cache like redis. Possible problem can be faced is the race conditions in distributed systems for the in-memory cache (Solvable with locks, but will affect performance).
### Chapter 5 (DESIGN CONSISTENT HASHING)
Used in distributed system. Hashing are usually used for load balancer to determine which server to direct the requests. Common hashing method depends on number of servers (server_id = Hash(server_name_or_key)%number of servers). Addition or removal of servers will affect the redirection of requests. Changing the server_id for each requests may result in cache misses or data redistribution.  

#### Consistent Hashing  
Consistent hashing can be achieved by hashing the server key and placed them on a 'ring'. Imagine the hash values are on a loop ring, 0 to 2^64-1 and back to 0. To determine which server the request should send to, we hash the requests (get a hash value) and we will search the nearest clockwise/right of the hash value. In this case, whenever we remove a server, the other requests that are not assigned to the server will not be affected. However, this is not perfect and still has another problems. If the hash values of the server distributions are not uniform, there may be an imbalance load on a particular server. This can non-uniform hash values distribution can also happen to the requests. Therefore, a new concept of virtual nodes are introduced. We will allocate multiple hash keys to a server and they are the virtual nodes. With more virtual nodes mapped onto the ring, the standard deviation of the hash space (spaces between each server nodes on the hash 'ring') become smaller. Hence it will be more balanced.
Algorithm
1. Declare the variables
    1. Virtual nodes count for each nodes
    2. Sorted Hashed Values (can be sorted list)
    3. Server Hashed Values to Real Node (Store the virtual nodes hash value to the real node)
    4. Set of server key (Optional, but you can check if the server key has already used by other server)
2. Add Server Node 
    1. For a fixed virtual nodes count (Let it be 100) 
    2. Create hash key for each virtual node of this server key (We need append a suffix of the virtual node index to the server key)
    3. Use binary search and insert into the sorted hashed values list
3. Remove Server Node
    1. For a fixed virtual nodes count (Let it be 100) 
    2. Create back the hash values for each virtual node of this server key
    3. Remove from the sorted list
4. Find the server based on request key/id
    1. hash the request key
    2. do a binary search and return the value right of your request key hash value
    3. use the server hashed values to real node mapping to find your real node
    4. return your real node which will be mapped to the request

### Chapter 6 (DESIGN A KEY-VALUE STORE)
1. For single server key value store, we can perform 2 optimisations (1. Data compressions, 2. Keep only frequently-used key in memory as hash map and the rest in disk)
2. CAP Theorem -> Consistency, Availability, Partition Tolerance
   1. Consistency -> (Eventual/Strong) Consistency refers to the data are the same for all the distributed servers
   2. Availability -> Refers to the client should get a response even if some of the nodes are down
   3. Partition Tolerance -> Refers the system continues to operate even if some packets are dropped
   4. No distributed systems are safe from network partition failure.
      1. When network partition failure occurs and we choose availability, we will sacrifice consistency as we need to return a response but it may not contain the most updated data
      2. When network partition failure occurs and we choose consistency, we will sacrifice availability as we need to wait to process the data to ensure all data are most updated
      3. If we priorities both availability and consistency, and network partition failure occurs, it is impossible for our system to be able to return every request a response with the most updated data when partition network failure occurs. The distributed system will not be able to run because it is not tolerate to network failure.
3. Data Partitions -> Allow data to placed on different servers (Challenges: Split data evenly and minimized data movements across server)
   1. To resolve the challenges, we can use consistent hashing. It allows us to easily increase the server counts and reduce data movements across servers.
   2. If we want a server to hold more data, we can add more virtual nodes for the physical server in consistent hashing.
4. Data Replication -> If we want to replicate the data in many servers, we can use the N numbers of unique physical nodes in the clockwise direction in consistent hashing
5. Consistency -> We need other servers to acknowledge the update from replicas
   1. Define N as number of replicas, W as the number of replicas to acknowledge write operation, R as the number of replicas to acknowledge read operation
   2. Note that acknowledging the operations DOES NOT MEAN it completed the write operation to update the values. It only means it received the write request and will update it. 
   3. If R = 1 and W = N, the system is optimized for a fast read.
   4. If W = 1 and R = N, the system is optimized for fast write.
   5. If W + R > N, strong consistency is guaranteed (Usually N = 3, W = R = 2).
   6. If W + R <= N, strong consistency is not guaranteed.
6. Inconsistency resolution
   1. Versioning -> Vector Clock (Server name, version) -> A technique to identify conflicts
   2. By storing the sequence of Vector Clock, we can keep track of the update sequence.
   3. However, there can be a conflict in vector clock when two independent update from two different servers occur
   4. Although it mention we can detect conflict, it did not mention how to resolve it.
7. Failure Detection -> Cannot be detected with simply just 1 node saying that a node is down
   1. Gossip Protocol
      1. Every node holds the node membership list
      2. Every node periodically send a heartbeat and state information of other nodes to random subset of nodes
      3. Every node that receive a heartbeat reply will update the latest timestamp to the list
      4. If the node also realised that the node the are indicated by other nodes have not replied, the node will also marked as 'down'.
8. Handling Failure
   1. Temporary Failure Management
      1. Sloppy Quorum
         1. In exchange of consistency for availability, where lesser number of read or quorum is required.
      2. Hinted Handoff
         1. Basically, if there is an write request by the system, other servers could help to store the updates.
         2. Read might not always work depending on the number of data replicas.
         3. Consider, we have R=3 with data replicated to 3 servers and 1 of the servers went down, other server does not have the data and cannot help to proceed the request when we require 3 read from different replicas.
   2. Permanent Failure Management
      1. Merkle Tree/ Hash Tree
9. Cassandra Database Design
   1.  Write Path: Client write key -> DB will store to cache first -> DB will flush cache to disk periodically or full -> DB disk store SST
   2.  Read Path: Client request key -> DB try read from cache -> If not found, DB checks bloom filter -> If bloom filter says yes, get the key from SST disk.
   3.  It adopted an append-only write operation
   4.  Each SSTable file contains an index. In the disk searching, it will need to scan through each SSTable file's index to find the key.
   5.  Periodically, compaction will be done to reduce the number of SSTable
   6.  Also, delete operation only marks to obsolete. 


### Chapter 7 (DESIGN A UNIQUE ID GENERATOR IN DISTRIBUTED SYSTEMS)
#### Requirements
1. IDs must be sortable by time and unique
2. IDs does not necessarily need to be increment by 1.
3. IDs should only contain numeric value and must be able to fit in 64 bits
4. The system should be able to generate up to 10,000 IDs per second.
#### Possible Designs
1. Having a singular `auto_increment` function that used in sql database. Multiple servers will request from it
   1. Pros: Easy to implement, it is sortable, numeric
   2. Cons: Single point of failure, Hard to scale
2. We can have different `k` servers and each having their own `auto_increment` function and the generated value will be multiplied by k and incremented by server index.
   1. Pros: Relatively easy to implement, more robust
   2. Cons: Hard to scale with data center, IDs cannot go up with time, It cannot scale well with the `k` value.
3. We can make use of UUID generator function in servers. UUID is relative hard to have collision. It is said to have collision in 100 years even if it generate 1 billions UUID every second
   1. Pros: We can scale the servers. It also relatively easy to implement by using the UUID generator. No need to synchronise across servers.
   2. Cons: It will not fit in 64-bit as UUID is 128 bits, IDs do not go up with time, IDs could be non-numeric.
4. Twitter Snowflake. We can split up the 64-bit IDs into different bits such that it will be largely scalable. For the first bit, we will preserve it in case of the need of negative value. From the second bit to 41st bit, it will be using the timestamp. From the 42nd bit to 46th bit, we can use the data center index. From the 47th bit to 51st bit, we can use the server index. Lastly, the remaining bit will be used for the same requests done in the same millisecond
   1. Pros: Scalable, it also fit in 64 bits, it can support up to the number 2 ^ 12 bits of concurrently requests in the same second, data center and server node. The timestamp has 41 bits which gives us 69 years
   2. Cons: Clock synchronisation (Not all servers may have the same clock), Might need to tune the remaining bits to give more bits for timestamp to allow more room for errors


### Chapter 8 (DESIGN A URL SHORTENER)
#### Requirements
1. Long URL need to be shorten
2. Shorten URL should only contains alphanumeric values.
3. Shorten URL can be be deleted or updated
4. Retain for 10 years
#### Shorten URL Length
It actually can be calculated based on how many URLs we wish to support at the end.  
Assume that shorten url can only have (0-9, A-Z, a-z) which is 10 + 26 + 26 = 62 characters, a length of 1 can represent 62 urls.  
Lets say we want to at least support 365 billion urls.  
We need to 62^n >= 365 billion urls  
Taking log both side -> log62 62^n = log62 365 billion -> n = log62 (365 billions), n is around 6.45.  
We should round up which n = 7.  
#### API Design
1. POST API for shorten a long url -> Takes in long url and return the shorten url
2. GET API for redirect the shorten url to the long url
   1. For redirection, it is important to use HTTP 301 (permanently moved) for browser caching. Browser caching means the browser will directly visit the longer url if it has visited the shorten url before. Pros: Reduce server load
   2. HTTP 302(temporarily moved) for dynamic url updating. This will allow the client always visit back the shorten url server. Pros: Allow tracking analytics
#### Storages
1. Usually, there will be a lot of URLs that will be shorten. Therefore it is not feasible to use a small database.
2. Commonly, we would be using a distributed datebase that stores id(pk), shortenURL, longURL

#### URL Shorten Design
1. There are many hash function. However, purely using hash function will usually not result in a shorter url.
2. Two main strategies
   1. Repeated hashing slicing
      1. Algorithm
         1. Check the longer url exists first with bloom filter. If exists, return the shorten url.
         2. Hash the longer url
         3. Slice to the shorter desired length
         4. Append to the home domain
         5. Check if the new shorter url exists -> We can use bloom filter. If exists, repeat step 2 to 4 with a new predefined string appended to the original url
         6. Store the mapping from longer url and shorten url in the database
      2. Pros
         1. Relatively simple and easy to understand
         2. Fixed length
      3. Cons
         1. May be very inefficient because we may need to keep rehashing
   2. Hashing + Base 62 Conversion
      1. Algorithm
         1. Check the longer url exists first with bloom filter. If exists, return the shorten url.
         2. Hash the longer url
         3. Note that hashed value are used in hexadecimal
         4. Make the output string shorter
            1. (Optional) Slice the hash string first (16^slice_length >= 67^7, which is approximately 12), depends on the speed of conversion to base 16
            2. We can convert from base 16 to base 10 and do a modulus of 62^ 7 (length of output string) then convert into base 62
         5. Append to the home domain
         6. Check if the new shorter url exists -> We can use bloom filter. If exists, repeat step 1 to 3 with a new predefined string appended to the original url
         7. Store the mapping from longer url to the shorten url in the database
      2. Pros
         1. It will be short and human-readable
         2. It can be fixed length only by doing the modulus

### Chapter 9 (DESIGN A WEB CRAWLER)
#### Requirements
1. Collect 1 billions page per month
2. Download HTML only
3. Store website up to 5 years
4. Ignore duplicated content
5. Easily extendable with more modules in the future to download more other contents like images and 
#### General Design
1. Cache DNS IP
2. Send request to download -> Allow smooth download in background
3. Hash the content -> Check if it is downloaded, we can have a bloom filter to check before searching database
4. Parse and check the desired content -> Reduce the content being saved
5. Send the write requests to the distributed databases.
6. Have a message queue to store the write requests
7. Extract more links form the html
8. Check link seen and filter some blacklisted sites
9. Not seen repeat from step 2
#### Links Exploration
BFS or DFS -> DFS may result in very deep depth search  
BFS is commonly used for web crawler because it allows a more balanced search for the sites.  
BFS does not consider the priority of downloading certain sites therefore we use other strategies like pagerank to give priority on more important webpages  
#### Important consideration
1. Politeness -> We do not want to overwhelm the web server that we are crawling which may crash their server. Generally, to avoid over-request on a certain host, we will add the download urls into a queue and such that the webpage are download according to the queue sequence. Additionally, we can add delay between two download tasks. We can also follow `robot.txt` and avoid crawling pages listed in `robot.txt`.
2. Priority -> We know we should prioritise more important pages. We can first have a prioritiser service where it will compute the importance of the webpage and place them into different queues based on their importance. The second politeness queue selector will randomly select one of the queue based on their importance before insert into their respectively host queue.
3. Freshness -> May need recrawl website periodically. We can recrawl based on web update history and in the order of their importance (and even more frequently).
4. Robustness
   1. We can have consistent hashing to distribute our crawling servers
   2. Save crawl states and data to enable fast recovery after crashing
   3. Exception handling
   4. Data Validation
5. Extensibility -> Able to parse more content for download by adding more system module after content seen
6. Avoid problematic issues
   1. There are many duplicated website where we should ignore download sites
   2. Endless loop of the site that may trap the web crawler
   3. Data noise like ads or spam url should be avoided
#### Performance Optimisation
1. Distributed crawls
2. Cache DNS -> Reduce the visit to DNS
3. Locality -> Crawling server that are nearer in actual location is always quicker
4. Short Timemout -> Reduce the number of blocking requests that may block the downloading process for a very extended time.

### CHAPTER 10 (DESIGN A NOTIFICATION SYSTEM)
#### Requirements
1. Support Push Notifications, Emails, Messages
2. Support iOS, Android and Desktop/Laptop
3. Soft Real Time
4. Notifications can be triggered via client/server side
5. Opt-out option should be made available
6. 10 million push notifications, 1 million SMS messages, 5 million emails

#### How to send notification
1. iOS -> Server send request, containing Device Token to Apple Push Notification (APN), and APN will send to the device
2. Android -> Server send request to Firebase Cloud Messaging (for non-China area)
3. Send SMS -> We can use Twilio 
4. Send Email -> We can use Sendgrid

#### Typical Flow
1. User send an requests on when they want to be notified
2. Server will collect the phone number, emails and device token when the user is registered to the notification server
3. When server receive the user's request, the server will first go to the cache to check if the user's information is already in the cache before visiting the database
4. Once received the user's information, the server will put the user's notification into its respective queue depending on its device and its notification type.
5. Workers will pull the user's notification from the queue and forward to the respective notification services (Apple Push Notification/ Firebase Cloud Message/ Twilio/ Sendgrid)
6. Based on the return code of the notification services, we can decide to retry and also monitor them. If there is too many failed notifications, we should notify the developer. 
7. Monitoring the queue is important as we may decide to increase the workers for processing the notifications if necessary.
8. Additionally, there can be further analysis on the notification interaction where we can allow request feedback from user's interaction with the notification.

#### Additional Consideration
1. Opt-in or out options -> Allow users to decide if they want to be notified
2. Rate Limiting -> Limit the number of notifications sent to users, to not overwhelm the users and causing them to opt-out of the notification

### CHAPTER 11 (DESIGN A NEWS FEED SYSTEM)

#### Requirements
1. Support mobile and web
2. User can publish posts and see her friends’ posts on the news feed page
3. Sorted by reverse chronological order
4. Maximum friends is 5000
5. 10 million DAU
6. Feed can contain images, files and text

#### Lets try design (Probably maybe wrong some part)
Every user can have up to 5000 friends. This means if we do not cache the user's news feed then we need to always run SQL to pull all their friends's posts and sort them by reverse chronological order. It is rather inefficient to do it every calls. Plus, we are not using it for recommendation. Ideally, we will store the items to be displayed in the database with pagination. So we will not be returning all the items in the new feeds, instead we will only send the requests by page. This will allow each request to be faster (less data sent across site).  

We also need to handle the popular Twitter celebrity problem. There may be a user that are connected with many users and post alot. This will cause the friends of 'celebrity' to not have the most updated news feed. What we can do here is to identify such celebrity and allow them to update their friends' new feeds whenever they upload a new post by prepending to the front of the list. In this sense, we do not need to make the celebrity's friends to pull all their friends' post to remake the news feeds.  

Since it contains images, files and text, it may be less efficient to store them in a relational daabases where multiple sql may be needed to get the different resources. We can choose a NoSQL databases to store such data together by store the encoded string of such data.  

Generally speaking, read should be more than write. Hence, we expect read operation to be fast.

Perhaps we can even add notification service to notify user on the new posts that are added to their news feed.

Client -> API Gateway (Load Balancer) -> Servers -> New Feeds Service -> Read Cache -> Databases
                                            |                                          /\
                                            \/                                          |
                                        Post Service -> Message Queue (This will update the friend's new feed)

#### Key Consideration
1. Load balancer with Web servers
2. Fanout Strategy
   1. Write Fanout ->  Update the friend's news feed when adding a post
   2. Read Fanout -> Run the SQL to get the news feed
   3. We prioritise read speed, hence we will use write fanout.
3. Caching Strategy
   1. News Feed Cache -> Consist of (post_id, user_id)
   2. User Cache -> Consist of user_id and user information
   3. Post Cache -> Consist of post_id and post objects
4. Media Content
   1. Stored in Content Delivery Service to allow fast retrieval
5. Database
   1. Graph Database

#### Typical Write Flow
1. A user send a request to add a post
2. The load balancer redistributes requests to web servers
3. Web servers call the news feed service to add a post
4. New Feed service will get friend ids from the database
5. Get friends info from the user cache. Filter out friends that wants to mute current user or hide posts to be display to friends selected by the user
6. Put the filtered friend list with current post_id into the message queue
7. Fanout workers will read from the message and store the (user_id, post_id) into news feed cache. In this manner, read time can retrieve the post_id list for a user_id.

#### Typical Read Flow
1. A user sends a request to retrieve her news feed
2. The load balancer redistributes requests to web servers
3. Web servers call the news feed service to fetch news feeds
4. News feed service gets a list post IDs from the news feed cache
5. A user’s news feed is more than just a list of feed IDs. It contains username, profile picture, post content, post image, etc. Thus, the news feed service fetches the complete user and post objects from caches (user cache and post cache) to construct the fully hydrated news feed
6. The fully hydrated news feed is returned in JSON format back to the client for rendering
7. Client will retrieve the media content from the JSON data received

#### Additional Consideration
1. Rate Limiting
2. Authentication
3. Fanout strategy
   1. Write Fanout
      1. Pros
         1. Read operations will be fast since we are updating news feed when it is required, and it is pre-computed. 
         2. Update can be near realtime.
      2. Cons
         1. If a users have many friends, fetching the friend list and generating news feeds for all of them are slow and time consuming. This is known as hotkey problem.
         2. For inactive users or those rarely log in, pre-computing news feeds waste computing resources
   2. Read Fanout
      1. Pros
         1. For inactive users or those who rarely log in, fanout on read works better because it will not waste computing resources on them
         2. Data is not pushed to friends so there is no hotkey problem
      2. Cons
         1. Fetching the news feed is slow as the news feed is not pre-computed
4. More caches to store more information. For example, we can have a 5 layer cache
   1. News Feed
   2. Content -> Consist of Popular content and Normal content cache
   3. Social Graph -> User Relationship Data
   4. Actions -> Store liked, replied and others
   5. Counters -> Store like counts, reply counts and others counts
5. Scaling database
   1. Vertical scaling vs Horizontal scaling
   2. SQL vs NoSQL
   3. Master-slave replication
   4. Read replicas
   5. Consistency models
   6. Database sharding

### CHAPTER 12 (DESIGN A CHAT SYSTEM)

#### Requirements
1. Support group chats and 1 on 1 chats
2. Support mobile and web app
3. Support 50 million DAU
4. Support up to 50 people in a group chat
5. Support online indicator feature and text message only
6. Support up to 100,000 characters message
7. End to end encryption not required
8. Support storing chat history forever

#### General Components
1. Notification Service
2. Multiple Device support -> Need authentication methods and authenticated device info should be stored in user info
3. User info -> Store phone number and blocked users
4. Group chat info stores user_ids, user_joining_logs and message logs (contains message, userid and timestamp)
5. 1 on 1 chats should generally store message info (contains message, userid and timestamp)
6. We can classify chats as two types - 1 on 1 and group chats
7. In this manner, we can store them in the same format. Just that 1 on 1 chats will only contain 2 or less participants

#### Core Concepts
1. Generally, chat systems are a message relay and storage system
2. There may be couple of messages that will be sent off by same users. It does not seem to be efficient to use HTTP to create TCP connection for each message sent by the user. Also, other users may want to send a message and the receiver may need to reestablish connection with the chat server to receive the message.
3. Why it is bad using HTTP for chat system?
   1. HTTP is client-initiated, meaning that chat server cannot forward to users
   2. HTTP is not designed to maintain long-term connection
4. Better strategy to handles relay problem OR server side initiated requests
   1. Polling -> Client will periodically ping the servers for new message
      1. Pros: Will be near realtime
      2. Cons: Servers may not be able to handle the requests when most of the time the request reply will be 'no'
   2. Long Polling ->  Keep connection with the server longer and server will only reply when there is a message
      1. Cons: It is hard for the server to know if the client is still connected. Still inefficient as polling
   3. Web Sockets -> Create a socket port for each user. On the server side, server can send the new message to the receiver's socket port anytime
      1. Pros: Do not need to repeatedly perform TCP handshakes
      2. Cons: This will introduce stateful components/services, which may increase complexity of the system
5. Probably would need to have a notification system -> Refer back to Chapter 10 (Can also use 3rd party notification system)
6. Databases Consideration (To be honest, it is vague in the book)
   1. Think about the data we need to store and read/write patterns
   2. Typically, two types of data
      1. Generic Data - User Profile, User Preference, User Friend List
      2. Chat History Data - Messages
   3. Generic Data does not require a lot of read and write. It would be easier to use relational database
   4. Messages wise we might need to process many data. However, most users are less likely to check on old messages. We can use key-value stores.
7. Message Schema
   1. 1 on 1 chat
      1. message_id (bigint)
      2. message_from (bigint)
      3. message_to (bigint)
      4. content (string)
      5. created_at (timestamp)
   2. group chat
      1. channel_id (bigint)
      2. message_id (bigint)
      3. userid (bigint)
      4. content (string)
      5. created_at (timestamp)
   3. message_id generation
      1. auto_increment (if we are not using nosql (key-value store))
      2. id generation (twitter snowflake, a little too overkill)
      3. local id generation (maintain id within same channel or one on one chat)
8. Service Discovery
   1. Allows all chat servers to register themselves
   2. Allows the backend servers to look for the best chat server geographically

#### Process Flow
1. Login
   1. User tries to authenticate itself to the server
   2. The load balancer will forward the login request to a server
   3. The server will look for the closest chat server geographically. The chat server is returned to the user
   4. User will connect to the chat server
2. Message Flow for 1 on 1 Or Group
   1. User send the message via the web socket connection to the chat server side
   2. Server will get the message id from the id generator
   3. Server will receive the message and place the message into the message sync queue for the **receiver**
   4. The message is stored in key-value store
   5. If the receiver is online, the server will forward the message to the receiver. Else, the server will send to the push notification server.
   6. Server will reply the user as processed
3. Message Synchronisation
   1. Each device maintain a max message_id for each chat
   2. To determine if we need to synchronise
      1. Check if the current logged-in user is the message recipient id
      2. Check if the max message-id in device is smaller than the max message-id in key-value store
4. Online Presence
   1. Considering a naive way, we force users to always send a information to indicate it is online and offline. In this manner, when user is disconnected, the user will never be marked as offline.
   2. Solution
      1. User will send an heartbeat periodically (around 5 seconds)
      2. Online presence server will update the latest online timestamp
   3. Pros: If users are disconnected from servers or online status message is dropped, the server will automatically marked it as offline
   4. Users can subscribe to the online presence server to obtain the desired user's status

### CHAPTER 13 (DESIGN A SEARCH AUTOCOMPLETE SYSTEM)
#### Requirements
1. Match keyword with the same prefixes
2. Return around 5 popular suggestions
3. Do not need to support spell check
4. Search results should only consist of English
5. Assume all results are in lowercase characters
6. 10 million DAU

#### Core Concepts
1. Data Aggregator Service
   1. To collect the query and accumulate the frequency
   2. Database wise we only need to store the query text and frequency
   3. For logging wise, it is smarter to store timestamp. In this manner, we can accumulate the frequency even if it is computed in batch manner.
2. Query Services
   1. This probably need to be very fast. Consider if it takes too long to return suggestion, user would have already finished typing.
   2. This rounds to a data structure -> Trie
      1. Trie is a famous data structure to search through prefix and find words with same prefixes
      2. However, we need to know that purely using trie is still too slow. 
      3. Consider, the length of prefix is `p` and the number of words under the prefix is `c`. Then, the time complexity would be `O(p) + O(clogc) + O(c)`.
      4. Because we need to sort by frequency for all the available candidates word.
      5. The better way to improve this is to cache the top k words at the trie node
      6. We need to discuss with the interviewer about the need of having a real-time popularity or the delayed version. 
      7. In common scenario, top frequent k words are very unlikely to be change for each prefix once it is built in the a short time.
      8. Hence, we can rebuilt it every two weeks.
   3. Flow
      1. User send search query to the load balancer
      2. Load balancer will route the request to the servers
      3. Servers will try to get the autosuggestion data from the Trie Cache
      4. If not found in Trie Cache, we will then fetch from Trie Database
   4. Optimisation
      1. AJAX Requests -> Asynchronisation Javascript And XML
         1. Send request to update part of the web page instead of reloading the whole web page
      2. Browser Caching -> Cache the autosuggestions result in the client site and set expiration time
         1. Set in `cache-control` headers of response
   5. More On Tries
      1. Create -> Every 2 weeks (or depending on the update frequency)
      2. Update
         1. Batch Manner -> Update every 2 week
         2. Individual Node Update -> This is very costly
      3. Delete -> Remove harmful/hateful words
3. Trie Cache
   1. Store the database trie weekly snapshot in memory
4. Trie Database
   1. Document Store
   2. Key-value Store
5. Data Sampling
   1. Instead of storing all the search queries, we can choose to only store the queries every N requests. This can save alot of processing power and storage.
6. Storage
   1. Sharding
      1. Simple Sharding -> Divide `a-z` into `a-i`, `j-r `and `s-z`. Level by level down `aa-ai`, `aj-ar` and `as-az`
      2. Distributed Sharding Based on Data 

### CHAPTER 14 (DESIGN YOUTUBE)
#### Requirements
1. Ability to upload and view video
2. Support mobile, web and smart TV
3. 5 Millions DAU
4. Average 30 minutes per user
5. Support internationally
6. Support most video formats and resolution
7. Encryption required
8. Max video size is 1GB
9. Leverage on Cloud Third Party Services

#### Core Concepts
1. Video -> Video Data and Meta Data
2. Video Transcoding
   1. Importance of Video Transcoding
      1. Raw video can takes up large amount of storage
      2. Many devices can only support certain types of video formats
      3. To ensure users watch high quality with smooth playback, allow user with higher bandwidth to load higher quality video and user with lower bandwidth to load lower quality video
   2. DAG (Direct Acyclic Graph) for video transcoding allowing more customisation of the video transcoding process
      1. Create custom thumbnails or auto generation of thumbnails
      2. Add watermark to video
      3. Perform inspection (Check of malformed video)
   3. Video Transcoding Architecture
      1. Preprocessor -> Process the configuration file to generate the DAG and cache video information in Temporary Storage
      2. DAG Scheduler -> Split parallisable tasks of the dag into different task queues
      3. Resource Manager
         1. Task Queue 
         2. Worker Queue
         3. Running Queue
         4. Task Scheduler -> Choose the optimal task/worker and instruct the worker to execute the task
      4. Task Workers -> Run the tasks
      5. Encoded Video
3. Video Uploading
   1. We can parallelise the two data
   2. Video Data Upload
      1. User will send the video data to the original storage server
      2. The original storage server will send the video to the Transcoding servers
      3. Transcoding server will process the video data into different formats for different devices
      4. Two steps in parallel, once completed
         1. Transcoded video will be stored in transcoded storage
            1. Transcoded video will also be sent to Content Delivery Network (CDN)
         2. Transcoded video completion info will be placed into a completion queue
            1. Completion queue worker will pull the info and update the metadata database
      5. API Server will inform the client on the status of the video upload 
   3. Video Metadata Upload (Run with video data upload concurrently)
      1. User will send the video meta data to the api server
      2. API server will upload the metadata to the metadata storage
4. Video Streaming
   1. Make use of streaming protocols 
      1. MPEG–DASH. MPEG stands for “Moving Picture Experts Group” and DASH stands for "Dynamic Adaptive Streaming over HTTP"
      2. Apple HLS. HLS stands for “HTTP Live Streaming”
      3. Microsoft Smooth Streaming
      4. Adobe HTTP Dynamic Streaming (HDS)
   2. Videos are streamed from nearest CDN
5. Error Handling
   1. Upload error: retry a few times
   2. Split video error: if older versions of clients cannot split videos by GOP alignment, the entire video is passed to the server. The job of splitting videos is done on the server-side. 
   3. Transcoding error: retry.
   4. Preprocessor error: regenerate DAG diagram.
   5. DAG scheduler error: reschedule a task.
   6. Resource manager queue down: use a replica.
   7. Task worker down: retry the task on a new worker.
   8. API server down: API servers are stateless so requests will be directed to a different API server.
   9. Metadata cache server down: data is replicated multiple times. If one node goes down, you can still access other nodes to fetch data. We can bring up a new cache server to replace the dead one
   10. Metadata DB server down:
       1.  Master down: Promote a slave to be the new master
       2.  Slave down: Direct the read to the other slaves

#### Speed Optimisation
1. Parallise video uploading
   1. Split video into smaller chunks and allow resumable videos when video upload fails
2. Place upload center nearer to users
3. More Parallisation
   1. Adding more message queues in the video transcoding download and video transcoding. These allows higher CPU ultilisation.
   2. Similarly we can do it for transcoded video completion and CDN video uploading 

#### Safety Optimisation
1. Pre-signed URL
   1. A temporary url that gives access to specific resources
2. Protect the videos
   1. Digital Rights Management Systems
   2. Encrypted Video
   3. Watermarking video

#### Cost-saving Optimisation
1. CDNs Cost
   1. Store popular video on CDNs and retrieve less popular from transcoded database
   2. Store long duration video on CDNS and process the short duration video on demand since it may be processed quickly
   3. Serve popular video that is specifically for the regions to the CDNs (in the region)
   4. Build your own CDNs

### CHAPTER 15 (DESIGN GOOGLE DRIVE)
#### Requirements
1. Upload, download file, sync file and notifications
2. Mobile and Web App
3. Supports any file type
4. File should be encrypted
5. File size limit is 10GB
6. 10 Million Daily Active Users

#### API Design
1. Upload
   1. Simple -> Small file we can simply just upload
   2. Resumable -> Large file we can make them resumable such that the whole file do not need to reupload due to connection dropped
2. Download
3. File Revision

#### Challenging Problems
1. Sync Conflicts -> Simplest way is first write win and allow the user to choose if they allow the overwrite from the server version [4][5]
   1. https://neil.fraser.name/writing/sync/ -> Merge the client's changes when the user stop typing and discard overlapped changes if there is a conflict
2. High Consistency Requirement -> Invalidate cache on read replicas when there is a database write
3. Download/Sync Flow -> Depending on the status of the user
   1. If the user is online, we will notify the user to pull the latest changes
   2. If the user is offline, we will store the changes to the cache and the user will pull from the cache when the user is online
   3. The pull process would be pulling the metadata from the metadata server and send the metadata to the block server for the latest changes update.
4. Notification Service -> WebSocket or Long Polling?/
   1. Do we need to real time update the client that the file is updated by other users or clients? The answer is No.
   2. Technically, there is not much purpose to use websocket since the user can check for updates periodically when the client is online
   3. Additionally, websocket is ideal for sending small packets, it is prone to connection drop and websocket does not support data compression like HTTP.
   4. Hence, it is more ideal to use long polling.

#### Smart Optimisation
1. Encryption and Storage Speed/Reliability -> Block Servers like S3
   1. Split data into blocks and encrypt them
   2. Only sync blocks that are changed/modified -> RSync
      1. Check the block's checksum with most updated block's checksum before determining if we need to sync the blocks
2. Parallize the requests on uploading/editing the file and the meta data of the file -> We separate them into two services and two systems so they can run concurrently.
   1. Add file metadata
      1. Client 1 sends a request to add the metadata of the new file
      2. Store the new file metadata in metadata DB and change the file upload status to “pending.”
      3. Notify the notification service that a new file is being added.
      4. The notification service notifies relevant clients (client 2) that a file is being uploaded.
   2. Upload files to cloud storage
      1. Client 1 uploads the content of the file to block servers
      2. Block servers chunk the files into blocks, compress, encrypt the blocks, and upload them to cloud storage
      3. Once the file is uploaded, cloud storage triggers upload completion callback. The request is sent to API servers
      4. File status changed to “uploaded” in Metadata DB
      5. Notify the notification service that a file status is changed to “uploaded.”
      6. The notification service notifies relevant clients (client 2) that a file is fully uploaded.